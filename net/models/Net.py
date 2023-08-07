import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import  conv
from torch.nn.modules.utils import _pair
import math
import numpy as np
from former import ImageTransformer


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

# dim%8==0
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return y.expand_as(x)

class CrossFeatureExtraction(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CrossFeatureExtraction,self).__init__()
        self.layer_s = nn.Sequential(nn.Conv2d(in_channel,out_channel//4,kernel_size=7,padding=3),nn.PReLU())
        self.layer_f = nn.Sequential(nn.Conv2d(in_channel,out_channel//4,kernel_size=5,padding=2),nn.PReLU())
        self.layer_t = nn.Sequential(nn.Conv2d(in_channel,out_channel//4,kernel_size=3,padding=1),nn.PReLU()) 
        self.layer_pw = nn.Sequential(nn.Conv2d(in_channel,out_channel//4,kernel_size=1),nn.PReLU())
        self.layer_down_2 = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),nn.Tanh())
        self.se_block = SE_Block(in_channel)

    def forward(self,x):
        x_s = self.layer_s(x)
        x_f = self.layer_f(x)
        x_t = self.layer_t(x)

        x_pw = self.layer_pw(x)
        # 3 * 4->将channel数再变回--
        # 是对x进行concate 或者变换完毕直接进行mix 不进行Mix 因为不同级
        x_total = torch.cat([x_s,x_f,x_t,x_pw],dim=1)

        # 使用3*3卷积来实现--
        output = self.layer_down_2(x_total+x)
        output = self.layer_down_2(output)
        att_coeff = self.se_block(x)
        # 使用残差结构
        output = output*att_coeff
        return output
        
class Channel_Sperate(nn.Module):
    def __init__(self, channel, ratio):
        super(Channel_Sperate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.GELU(),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.GELU(),
            nn.Linear(channel , channel ,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        out = self.fc(y).view(b,c,1,1) 
        return out
    


#Encoder-Decoder  
class SplitRGB_Module(nn.Module):
    def __init__(self,input_nc = 3,output_nc = 3,ngf = 32,use_dropout =False,padding_type = 'reflect'):
        super(SplitRGB_Module, self).__init__()
        self.down_1 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(input_nc,ngf,kernel_size=7,padding=0),nn.ReLU(True))
        self.down_2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride = 2,padding=1),nn.ReLU(True))
        self.down_3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride = 2,padding=1),nn.ReLU(True))
        self.DOWNUP_2 = CrossFeatureExtraction(ngf*2,ngf*2)
        self.DOWNUP_1 = DehazeBlock(default_conv,ngf*4,3)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=3,stride = 2,padding=1,output_padding=1),nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=3,stride=2,padding=1,output_padding=1),nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(ngf,output_nc,kernel_size=7,padding=0),nn.Tanh())
        self.mix1 = Mix(m = -1)
        self.mix2 = Mix(m = -0.6)
        self.DOWNUP_2 = DehazeBlock(default_conv,ngf*2,3)
        self.DOWNUP_1 = DehazeBlock(default_conv,ngf*4,3)

    def forward(self,input):
        x_down1 = self.down_1(input)
        x_down2 = self.down_2(x_down1)
        x_down3 = self.down_3(x_down2)
        # 经过下采样2 的feature_map 连接到上采样2
      
    
        # x_out_mix_ = self.mix1(x_down3_3,x_down3)
        x_up1 = self.up1(x_down3)
        x_out_mix = self.mix2(x_down2,x_up1)
        x_up2 = self.up2(x_out_mix)
        x_up3 = self.up3(x_up2)

        return x_up3


class Channel_Sperate(nn.Module):
    def __init__(self, channel, ratio):
        super(Channel_Sperate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.GELU(),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.GELU(),
            nn.Linear(channel , channel ,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        out = self.fc(y).view(b,c,1,1) 
        return out


class TPSM(nn.Module):
    def __init__(self):
        super(TPSM, self).__init__()
        self.dim = 64
        self.conv =  nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,padding=1),
            nn.Tanh()
            )
        # self.dim = 64
        conv = nn.Conv2d
        pre_process = [conv(3, self.dim, kernel_size = 3,padding=1 )]
        post_precess = [
            conv(self.dim, self.dim, kernel_size = 3,padding=1),
            conv(self.dim, 3, kernel_size =3 ,padding=1)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)
        self.PreModule = CrossFeatureExtraction(self.dim, self.dim)
        self.Post_Fusion = nn.Sequential(
            nn.Conv2d(in_channels = 9,out_channels = 6,kernel_size=3,stride=1,padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels = 6,out_channels = 3,kernel_size= 3,stride=1,padding=1)
        )
        self.Encoder_Decoder = SplitRGB_Module()
        self.trans = ImageTransformer(patch_size=4, in_channels=3, embed_dim=64, num_heads=4, feedforward_dim=256, num_layers=4, dropout=0.1)
        self.DehazeBlock = DehazeBlock(default_conv,self.dim,3)
        self.mix = Mix(m = -0.6)


    def forward(self,x):
        trans_map = self.trans(x)
        trans_map = self.conv(trans_map)
        conv_x = self.conv(x)
        ppre_x = self.pre(conv_x)
        mid_x = self.PreModule(ppre_x)
        pre_x = self.post(mid_x)
        pre_x = self.conv(pre_x)
        mid_x = pre_x + conv_x
        fea_x = self.Encoder_Decoder(mid_x)
        fea_x = self.pre(fea_x)
        fea_x = self.PreModule(fea_x)
        fea_x = self.post(fea_x)
        output = x*trans_map + mid_x + fea_x
        output = self.conv(output)
        return output