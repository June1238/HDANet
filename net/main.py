import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
import sys
from models import Net
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from metrics_ import psnr, ssim
warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16
from Losses.Msssim import msssim,ssim_Loss,LossNetwork

print('log_dir :', log_dir)
print('model_name:', model_name)
print('alpha: ',opt.alpha)

result_save_name=''
result_save_path=''

models_ = {
    'myNet': Net.TPSM()
}
loaders_ = {
    #'DHaze_NYU_train':NYU_train_loader,
    #'DHaze_NYU_test':NYU_test_loader,   
    'nh_train': NH_train_loader,
    'nh_test': NH_test_loader,
    'data_2021_train': data_2021_train_loader,
    'data_2021_test': data_2021_test_loader
    #'its_train': ITS_train_loader,
    #'its_test': ITS_test_loader
}

start_time = time.time()
T = opt.steps
device='cuda' if torch.cuda.is_available() else 'cpu'
 
def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion, result_save_path):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    with open(result_save_path,'a+') as f:
        f.write("Start Train.\n")

    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        out = net(x)
        loss = 0
        
        loss_1 = criterion[0](out,y)
        loss_2 = criterion[1](out, y)
        loss_3 = msssim(out,y)
        loss_4 = ssim_Loss(out,y)
        loss = loss_1
        print("loss is ：",loss)
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            
            if ssim_eval > max_ssim or psnr_eval > max_psnr :
                save_name = './trained_models/CXYNet_Data2021_803_PSTM_'+str(step)+'.pk'
                torch.save({
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict()
                 }, save_name)

            max_ssim = max(max_ssim, ssim_eval)
            max_psnr = max(max_psnr, psnr_eval)
            with open(result_save_path,'a+') as f:
                str1 = 'STEP:'+str(step)+' SSIM:'+str(ssim_eval)+' PSNR:'+str(psnr_eval)+'\n'
                f.write(str1)
            print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
            
        

    with open(result_save_path,'a+') as f:
        f.write(f'\nmax_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}\n')
    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = net(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        if step % 1000 == 0:
            torchvision.utils.save_image(targets,f'./test_pics/{result_save_name}/{step}_{i}_gt.png')
            torchvision.utils.save_image(pred,f'./test_pics/{result_save_name}/{step}_{i}_pr.png')
        ssims.append(ssim1)
        psnrs.append(psnr1)
    torch.cuda.empty_cache()
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    # criterion.append(msssim().to(opt.device))
    criterion.append(LossNetwork().to(opt.device))
    # criterion.append(ssim_Loss.to(opt.device))
        
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    result_save_name='CXYNet807_3'
    result_save_path = '.././Result_records/'+result_save_name+'.txt'
    if not os.path.exists(f'./trained_models/{result_save_name}/'):
        os.mkdir(f'./trained_models/{result_save_name}/')
    if not os.path.exists('./test_pics'):
        os.mkdir('./test_pics')
    if not os.path.exists(f'./test_pics/{result_save_name}/'):
        os.mkdir(f'./test_pics/{result_save_name}/')
    train(net, loader_train, loader_test, optimizer, criterion,result_save_path)
