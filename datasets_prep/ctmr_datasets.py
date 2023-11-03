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

class CBCTDataset(data.Dataset):
    def __init__(self, path, size=None,test=None):
        self.img = os.listdir(path)
        self.size = size
        self.path = path
        self.test = test

    def __getitem__(self, item):
        imagename = os.path.join(self.path, self.img[item])

        npimg = np.load(imagename)
        npimg = npimg.astype(np.float32)

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs * 255

        if self.test==True:
            window_size=9
            bilater_random1 = 75
        else:
            window_size = random.randint(1,15)
            bilater_random1 = random.randint(60, 120)
        nplabs = cv2.bilateralFilter(nplabs, window_size, bilater_random1, bilater_random1)

        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs * 255
        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        npimg = npimg

        return npimg, nplabs

    def __len__(self):
        size = len(self.img)
        return size


