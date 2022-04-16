import math
#import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#from skimage import metrics

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
            PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_ssim(img, imclean, data_range):
    Img = img.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        if Img.shape[1] == 1:
            SSIM += structural_similarity(np.squeeze(Iclean[i, :, :, :]), np.squeeze(Img[i, :, :, :]), 
                                          data_range=data_range, multichannel=False)
        else:
            SSIM += structural_similarity(np.squeeze(Iclean[i, :, :, :]).transpose(1, 2, 0), np.squeeze(Img[i, :, :, :]).transpose(1, 2, 0), 
                                          data_range=data_range, multichannel=True)
    return SSIM / Img.shape[0]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
