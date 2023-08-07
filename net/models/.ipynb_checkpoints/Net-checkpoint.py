import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import  conv
from torch.nn.modules.utils import _pair
import math

class SplitRGB_Module(nn.Module):
    def __init__(self,input_nc = 1,output_nc = 1,ngf = 32,use_dropout =False,padding_type = 'reflect'):
        super(SplitRGB_Module, self).__init__()
        #使用三层Unet分别进行--
        self.down_1 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(input_nc,ngf,kernel_size=7,padding=0),nn.ReLU(True))
        self.down_2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride = 2,padding=1),nn.ReLU(True))
        self.down_3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride = 2,padding=1),nn.ReLU(True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=3,stride = 2,padding=1,output_padding=1),nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=3,stride=2,padding=1,output_padding=1),nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(ngf,output_nc,kernel_size=7,padding=0),nn.Tanh())

        # self.flat_2 = nn.Sequential(nn.Conv2d(output_nc*2,3,kernel_size=3,stride=1,padding=1))
    def forward(self,input):
        x_down1 = self.down_1(input)
        x_down2 = self.down_2(x_down1)
        x_down3 = self.down_3(x_down2)

        x_up1 = self.up1(x_down3)
        x_up2 = self.up2(x_up1)
        x_up3 = self.up3(x_up2)

        return x_up3

#求取各个channel的自适应权重--
class Channel_Sperate(nn.Module):
    def __init__(self, channel=3, ratio = 2):
        super(Channel_Sperate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #使用的都是Linear层 没有卷积层Conv2d层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel , channel * 3,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y) 
        return y

class TotalM(nn.Module):
    def __init__(self,in_channels = 3,out_channels = 3):
        super(TotalM, self).__init__()
        self.PreModule = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.Sigmoid()
        )
        self.Post_Fusion = nn.Sequential(
            nn.Conv2d(in_channels = 9,out_channels = 6,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 6,out_channels = 3,kernel_size= 3,stride=1,padding=1)
        )
        self.R_Extract = SplitRGB_Module()
        self.G_Extract = SplitRGB_Module()
        self.B_Extract = SplitRGB_Module()
        self.ChanelAdaption = Channel_Sperate()
        
        self.flat_R = nn.Sequential(nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.flat_G = nn.Sequential(nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.flat_B = nn.Sequential(nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.alpha = 0.3
    def forward(self,x):
        pre_x = self.PreModule(x)
        R_M = self.R_Extract(pre_x[:, 0:1, :, :])*x[:, 0:1, :, :]
        G_M = self.G_Extract(pre_x[:, 1:2, :, :])*x[:, 1:2, :, :]
        B_M = self.B_Extract(pre_x[:, 2:3, :, :])*x[:, 2:3, :, :]

        R_M = self.flat_R(R_M)
        G_M = self.flat_G(G_M)
        B_M = self.flat_B(B_M)
        output = torch.cat((R_M,G_M,B_M),dim=1)
        y_w = self.ChanelAdaption(pre_x)
        b,c,_,_ = output.size()
        y_w = y_w.view(b,c,1,1)
        output = output*y_w.expand_as(output)

        output = self.Post_Fusion(output)
        output = output*x+pre_x

        return output


