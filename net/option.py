import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore') 

parser=argparse.ArgumentParser()
parser.add_argument('--steps',	type=int,default=35000) #训练次数
parser.add_argument('--device',	type=str,default='Automatic detection')  #训练设备
parser.add_argument('--resume',	type=bool,default=False)
parser.add_argument('--eval_step',type=int,default=250) 
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') 
parser.add_argument('--alpha', default=0.95, type=float, help='params for Orgin_Negative_Image') #学习率
parser.add_argument('--model_dir',type=str,default='./trained_models/') #存放模型的目录
parser.add_argument('--trainset',type=str,default='data_2021_train') #训练集
parser.add_argument('--testset',type=str,default='data_2021_test')  #测试集
parser.add_argument('--net',type=str,default='myNet') #网络架构
parser.add_argument('--gps',type=int,default=3,help='residual_groups') #groups 使用残差网络组
parser.add_argument('--blocks',type=int,default=5,help='residual_blocks') #残差块
parser.add_argument('--bs',type=int,default=16,help='batch size') #批量数据大小
parser.add_argument('--crop',action='store_true') #是否裁剪
parser.add_argument('--crop_size',type=int,default=360,help='Takes effect when using --crop ') #裁剪尺寸
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss') #感知损失函数

opt=parser.parse_args([]) #参数列表为空

opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+"AAAI"
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)

#创建各个文件--
if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
