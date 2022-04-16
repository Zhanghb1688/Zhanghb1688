import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from PIL import Image
from torchvision import transforms

class Dataset(object):
    def __init__(self, images_dir):#, patch_size):#, jpeg_quality, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))

    def __getitem__(self, idx):

        label = Image.open(self.image_files[idx])#.convert('L')
        label = transforms.RandomCrop(160)(label)
        label = transforms.ToTensor()(label)
        #label = transforms.RandomErasing(p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0), value=0.0, inplace=False)(label)

        return label#input, 

    def __len__(self):
        return len(self.image_files)
    
class Dataset1(object):
    def __init__(self, images_dir):#, patch_size):#, jpeg_quality, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))

    def __getitem__(self, idx):
        label = Image.open(self.image_files[idx])#.convert('L')

        return transforms.ToTensor()(label)#input, 

    def __len__(self):
        return len(self.image_files)
