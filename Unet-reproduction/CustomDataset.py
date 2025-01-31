"""
在这个类中我自定义了我的dataset类，用于自动调用某一张图片以及其对应的掩码，在我的代码中，我主要面向的是Kaggle中车俩分割的项目，
网址为https://www.kaggle.com/c/carvana-image-masking-challenge
"""
from idlelib.browser import transform_children
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.data.augment import ToTensor
import torch

class Carvana(Dataset):
    def __init__(self,image_dir:str,mask_dir:str,transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transformer = transform
        self.image = []
        for i in Path(image_dir).iterdir():
            self.image.append(str(i.name).strip(i.suffix))

    def __len__(self):
        return len(self.image)

    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir,self.image[idx] + '.jpg')
        mask_path = os.path.join(self.mask_dir,self.image[idx] + '_mask.gif')
        img = np.array(Image.open(img_path).convert("RGB"))    #Image默认是四通道格式，需要转换
        mask =np.array(Image.open(mask_path).convert("L"))
        mask = mask/255
        if self.transformer is not None:
            augmentation = self.transformer(image=img,mask=mask)
            img = augmentation['image']
            mask = augmentation['mask']     #这里尚未写transform的代码

        return img,mask






if __name__ == '__main__':
    trans = A.Compose([A.Resize(360,480),A.Normalize([0,0,0],[1,1,1],max_pixel_value=255),ToTensorV2()])
    a = Carvana("dataset/train","dataset/train_masks",trans)
    x,y = a.__getitem__(5)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            print(y[i,j].item(),end=' ')
        print()

