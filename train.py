import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import BasicBlock, RatUNet
from data import Dataset, Dataset1
from utils import AverageMeter,batch_PSNR, batch_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="REDNet")
#parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
#parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
#parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--image_path', type=str, default='/data/Set5')
parser.add_argument('--images_dir', type=str, default='/data/train1')
#parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
#parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.95e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def main():
    best_prec = 0
    best_ssim = 0

    # Load dataset
    print('Loading dataset ...\n')

    images_dir = os.getcwd() + opt.images_dir
    image_path = os.getcwd() + opt.image_path
    dataset_train = Dataset(images_dir)#, opt.patch_size, opt.noiseL)
    dataset_val = Dataset1(image_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    model = RatUNet(BasicBlock, 64)
    criterion = nn.MSELoss(reduction='sum')#.L1Loss(reduction='sum')
    #criterion = CharbonnierLoss()
    # Move to GPU
    #device_ids = [0]
    model = model.to(device)
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr

    step = 0

    sgdr = CosineAnnealingLR(optimizer, opt.epochs * len(loader_train), eta_min=0.0, last_epoch=-1)
     for epoch in range(opt.epochs):

        epoch_losses = AverageMeter()
        print('learning rate %f' % param_group["lr"])
        # train
        for i, data in enumerate(loader_train):

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = model(imgn_train)

            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            epoch_losses.update(loss.item(), len(img_train))         
            loss.backward()
            optimizer.step()
            sgdr.step()
            
            # results
            model.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            if i % 20 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch+1, i+1, len(loader_train), epoch_losses.avg, psnr_train))#loss.item()

            step += 1
        ## the end of each epoch
        model.eval()

        if epoch +1 == opt.epochs:
            torch.save(model, os.path.join(opt.outf, 'model.pth'))
         
if __name__ == "__main__":
    main()
