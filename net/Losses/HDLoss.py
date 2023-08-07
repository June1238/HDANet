import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_ImageMap(height,width,Image_hazy,Image_clear):
    image_clear = Image_clear.to(torch.float32).to(device)
    image_hazy = Image_hazy.to(torch.float32).to(device)
    shape_1,shape_2,shape_3,shape_4 = image_clear.shape[0],image_clear.shape[1],image_clear.shape[2],image_clear.shape[3]
    
    image_sub = torch.abs(image_clear-image_hazy).to(device)
    image_diff = torch.sum(torch.abs(image_clear-image_hazy),dim = [2,3]).to(device)
    mask = torch.zeros(shape_1,shape_2,shape_3,shape_4).to(torch.float32).to(device)
    map_coeff = torch.zeros(shape_1,shape_2,shape_3,shape_4).to(torch.float32).to(device)
    i = 0
    j = 0
    while True:
        sub_temp = torch.sum(image_sub[:,:,i:i+height,j:j+width],dim = [2,3])/image_diff
        sub_temp = sub_temp.reshape((shape_1,shape_2,1,1)).expand((-1,-1,height,width))
        mask[:,:,i:i+height,j:j+width] += sub_temp
        map_coeff[:,:,i:i+height,j:j+width] += 1
        j = j + width//2
        if j+width > shape_4:
            j = 0
            i = i + height//2
            if i+height > shape_3:
                break

    return mask/map_coeff

class HDLoss(nn.Module):
    def __init__(self,ablation = False):
        super(HDLoss,self).__init__()
        self.l1 = nn.L1Loss()
        self.ab = ablation

    def forward(self,a,p,n):
        loss = 0
        width,height = a.shape[2]//4,a.shape[3]//4
        mask = cal_ImageMap(width,height,a,n).to(device)
        if not self.ab:
            a,p = mask*a,mask*p
        loss = self.l1(a,p)

        return loss