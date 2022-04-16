import os
import argparse
import glob

import torch

from torch.autograd import Variable
from utils import batch_PSNR, batch_ssim
import torch.backends.cudnn as cudnn

from PIL import Image
from torchvision import transforms


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="RatUNet_Test")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='data/Set12', help='test on Set12 or Set68 Urban100 CBSD68 McMaster')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
opt = parser.parse_args()


def main():

    logs_dir = os.getcwd()
    # Build model
    print('Loading model ...\n')
    
    model = torch.load(os.path.join(logs_dir, 'logs/model.pth'))#
    model = model.to(device)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(logs_dir, opt.test_data, '*'))
    files_source.sort()
    # process data
    psnr_val = 0
    ssim_val = 0
    for f in files_source:
        # image
        label = Image.open(f).convert('L')
        box = (label.size[1] -  label.size[1]%8, label.size[0] -  label.size[0]%8)
        label = transforms.RandomCrop(box)(label)
        label = transforms.ToTensor()(label)
        img_val = torch.unsqueeze(label, 0)

        torch.manual_seed(64)
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.test_noiseL/255.)
            
        imgn_val = img_val + noise

        img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())

        with torch.no_grad():
            out = model(imgn_val)#, noise1)
            out_val = torch.clamp(out, 0., 1.)
        psnr = batch_PSNR(out_val, img_val, 1.)    
        ssim = batch_ssim(out_val, img_val, 1.)

        psnr_val += psnr
        ssim_val += ssim
        print("图像文件名: %s, PSNR_val: %.4f  SSIM_val: %.4f" % (f, psnr, ssim))    
    psnr_val /= len(files_source)
    ssim_val /= len(files_source)
    print("PSNR_val: %.4f  SSIM_val: %.4f" % (psnr_val, ssim_val))


if __name__ == "__main__":
    main()
