#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:34:00 2021

@author: zhb
"""


from PIL import Image, ImageDraw, ImageFont
import os

rootdir = os.getcwd() + '/train1'
rootdir1 = os.getcwd() + '/train1'
dirs = os.listdir(rootdir)
dirs.sort()
for filename in dirs:

        print('filename is :' + filename)
        currentPath = os.path.join(rootdir, filename)
        currentPath1 = os.path.join(rootdir1, filename)
        print('the fulll name of the file is :' + currentPath)

        im = Image.open(currentPath)
        #进行上下颠倒
#        out = im.transpose(Image.FLIP_TOP_BOTTOM)
#        newname = currentPath.replace('.png','_topbo.png')
#        out.save(newname)
        #进行左右颠倒
#        out =im.transpose(Image.FLIP_LEFT_RIGHT)
#        newname = currentPath.replace('.png','_lr.png')
#        out.save(newname)
         #进行旋转90
        out = im.transpose(Image.ROTATE_90)
        newname = currentPath.replace('.png','_rot90.png')
        out.save(newname)
         #进行旋转180
        out = im.transpose(Image.ROTATE_180)
        newname = currentPath.replace('.png','_rot180.png')
        out.save(newname)
         #进行旋转270
        out = im.transpose(Image.ROTATE_270)
        newname = currentPath.replace('.png','_rot270.png')
        out.save(newname)
        #将图片左右翻转以后旋转90
#        out =im.transpose(Image.FLIP_LEFT_RIGHT)
#        out = im.transpose(Image.ROTATE_90)
#        newname = currentPath.replace('.png','_lrrot90.png')
#        out.save(newname)
#        #将图片左右翻转以后旋转270
#        out =im.transpose(Image.FLIP_LEFT_RIGHT)
#        out = im.transpose(Image.ROTATE_270)
#        newname = currentPath.replace('.png','_lrrot270.png')
#        out.save(newname)
        #将图片重新设置尺寸
#        #out= out.resize((1280,720))
#        out = im.transpose(Image.TRANSVERSE)
#        newname = currentPath1.replace('.png','_trvs.png')
#        out.save(newname)
#        
#        #out = im.transpose(Image.TRANSVERSE)
#        out = out.transpose(Image.FLIP_LEFT_RIGHT)
#        newname = currentPath1.replace('.png','_trvslr.png')
#        out.save(newname)
#        
#        out = im.transpose(Image.TRANSVERSE)
#        out = out.transpose(Image.FLIP_TOP_BOTTOM)
#        newname = currentPath1.replace('.png','_trvsbt.png')
#        out.save(newname)