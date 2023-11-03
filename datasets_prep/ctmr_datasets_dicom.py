# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import random

import torch
from torch import nn
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
import torch.utils.data as data
import numpy as np
import lmdb
import os
import pydicom as dicom
import io
from PIL import Image
import cv2
#from skimage import transform
from torchvision import transforms
import matplotlib.pyplot as plt

def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)



class PairedCTMRDataset(data.Dataset):
    def __init__(self, path, name='ctmr', train=True, transform=None):
        self.img = os.listdir(path)  # [:4]
        self.shape = [220, 240]
        self.size = 192
        self.path = path

    def __getitem__(self, item):
        imgcase, randcase = divmod(item, 4)
        randx = np.random.randint(0, self.shape[0] - self.size)
        randy = np.random.randint(0, self.shape[1] - self.size)
        imagename = os.path.join(self.path, self.img[imgcase])

        itkimg = sitk.ReadImage(imagename)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)

        npimg99 = np.percentile(npimg, 98)
        npimg[npimg > npimg99] = npimg99

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs*255

        fft = False
        if fft == True:
            nplabs = cv2.GaussianBlur(nplabs,(3,3),0)


        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs*255


        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        
        nplabs = nplabs[randx:randx + self.size, randy:randy + self.size]





        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg = npimg.astype(np.float32)[randx:randx + self.size, randy:randy + self.size]


        resize = transforms.Resize([64, 64])

        nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs

    def __len__(self):
        size = len(self.img) * 4
        return size



class CBCTDataset(data.Dataset):
    def __init__(self, path, name='ctmr', train=True, transform=None):
        self.img = os.listdir(path)  # [:4]
        self.shape = [220, 240]
        self.size = 192
        self.path = path

    def __getitem__(self, item):
        imgcase, randcase = divmod(item, 4)
        imagename = os.path.join(self.path, self.img[imgcase])

        itkimg = dicom.dcmread(imagename)
        npimg = itkimg.pixel_array  # Z,Y,X,220*240*1
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)

        npimg99 = np.percentile(npimg, 97)
        npimg[npimg > npimg99] = npimg99

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs*255

        fft = False
        if fft == True:
            nplabs = cv2.GaussianBlur(nplabs,(3,3),0)


        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs*255


        #x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        #y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        #absX = cv2.convertScaleAbs(x)
        #absY = cv2.convertScaleAbs(y)
        #nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        nplabs = np.uint8(nplabs)
        nplabs =  cv2.Canny(nplabs, 100, 200)

        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([64, 64])

        nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs

    def __len__(self):
        size = len(self.img) * 4
        return size

class CBCTDataset_sobel(data.Dataset):
    def __init__(self, path, size = None,test=None):
        self.img = os.listdir(path)  # [:4]
        self.size = size
        self.path = path

    def __getitem__(self, item):

        imagename = os.path.join(self.path, self.img[item])

        itkimg = dicom.dcmread(imagename)
        npimg = itkimg.pixel_array  # Z,Y,X,220*240*1
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)
        #print(npimg.max())

        npimg99 = np.percentile(npimg, 97)
        npimg[npimg > npimg99] = npimg99

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs*255

        fft =True
        if fft == True:
            gau_random = random.randint(1,13)
            bilater_random1 = random.randint(60,120)

            nplabs = cv2.bilateralFilter(nplabs,gau_random,bilater_random1,bilater_random1)



        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs*255

        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        beta_random = random.random()
        #bright_range = random.randrange(0,5)
        #nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        nplabs = cv2.addWeighted(absX, beta_random, absY, 1-beta_random, 0)

        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs

    def __len__(self):
        size = len(self.img)
        return size
        

class CBCTDataset_test(data.Dataset):
    def __init__(self, path, size = None,test=None):
        self.img = os.listdir(path)  # [:4]
        self.shape = [220, 240]
        self.size = size
        self.path = path

    def __getitem__(self, item):

        imagename = os.path.join(self.path, self.img[item])

        npimg = np.load(imagename)
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs*255

            
        nplabs = cv2.bilateralFilter(nplabs,9,75,75)


        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs*255


        nplabs = np.uint8(nplabs)

        #canny_random = 140
        #nplabs =  cv2.Canny(nplabs, canny_random, 190)


        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)



        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs

    def __len__(self):
        size = len(self.img)
        return size
