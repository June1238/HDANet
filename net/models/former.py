import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size # 小块的大小，如(4, 4)
        self.in_channels = in_channels # 输入通道数，如3
        self.embed_dim = embed_dim # 嵌入向量的维度，如64

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2) 
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.feedforward_dim = feedforward_dim # 前馈神经网络的隐层维度，如256
        self.dropout = dropout 
        self.a = 16

        
        self.attn = nn.MultiheadAttention(2*embed_dim ,num_heads ,dropout=dropout,batch_first=True)

       
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim ,feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim ,embed_dim)
        )
       
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.linear_Q = nn.Linear(embed_dim, 2*embed_dim)
        self.linear_attn = nn.Linear(2*embed_dim, embed_dim)
        self.conv_K = nn.Conv1d(embed_dim, 2*embed_dim, kernel_size=1, stride=self.a)
        self.conv_V = nn.Conv1d(embed_dim, 2*embed_dim, kernel_size=1, stride=self.a)
        


    def forward(self ,x):

        Q = self.linear_Q(x)
        # 变换K
        K = self.conv_K(x.transpose(1, 2)) # b*n*m -> b*(2n)*(m/a)
        K = K.transpose(1, 2) # b*(m/a)*(2n)
        # 变换V
        V = self.conv_V(x.transpose(1, 2)) # b*n*m -> b*n*(m/a)
        V = V.transpose(1, 2) # b*(m/a)*n
        
    
        attn_out ,_ = self.attn(Q ,K ,V) # attn_out的形状为(batch_size ,seq_len ,embed_dim)，如(1 ,64 ,64)
       
        attn_out = self.linear_attn(attn_out)
        

        attn_out = F.dropout(attn_out ,p=self.dropout) # 对attn_out进行dropout
        x = x + attn_out # 残差连接
        x = self.ln1(x) # 层归一化

        # 前馈神经网络
        ffn_out = self.ffn(x) # ffn_out的形状为(batch_size ,seq_len ,embed_dim)，如(1 ,64 ,64)
        ffn_out = F.dropout(ffn_out) # 对ffn_out进行dropout
        x = x + ffn_out # 残差连接
        x = self.ln2(x) # 层归一化

        return x

class ImageTransformer(nn.Module):
    def __init__(self ,patch_size ,in_channels ,embed_dim ,num_heads ,feedforward_dim ,num_layers ,dropout):
        super().__init__()
        self.patch_size = patch_size 
        self.in_channels = in_channels 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.feedforward_dim = feedforward_dim 
        self.num_layers = num_layers 
        self.dropout = dropout 

        self.patch_embed = PatchEmbedding(patch_size ,in_channels ,embed_dim)

        
        # Local Perception Unit
        self.LPU = DWConv(in_channels, in_channels)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim ,num_heads ,feedforward_dim ,dropout) for _ in range(num_layers)
        ])

        self.reconstruct = nn.ConvTranspose2d(embed_dim ,in_channels ,kernel_size=patch_size ,stride=patch_size)
 
    def forward(self, x):
        c = x.shape[1]
        h = x.shape[2] 
        w = x.shape[3]
       
        x = self.LPU(x) + x
        x = self.patch_embed(x) 
       
        for layer in self.encoder_layers:
            x = layer(x) 

        x = x.transpose(1 ,2) 
        x = x.reshape(x.shape[0] ,x.shape[1] ,-1 ,self.patch_size) 
        x = self.reconstruct(x) 

        return x.reshape(x.shape[0] ,x.shape[1],h,w)