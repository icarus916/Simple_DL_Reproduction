import torch
import  albumentations as A
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from evaluate import PA
model = torch.load("models/unet_model19")
trans = A.Compose([A.Resize(height=160,width=240),A.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
image_path = r"D:\Academic Work\Paper Reproduction\Unet-reproduction\dataset\train\0ce66b539f52_03.jpg"
image = np.array(Image.open(image_path))
image = trans(image=image)['image'].to("cuda")
image = torch.unsqueeze(image,0)
out = model(image)
out[out>=0] = 255
out[out<=0] = 0
out = np.array(out.detach().cpu(),dtype=np.uint8)
out = np.squeeze(out,(0,1))
cv.imshow("cam",out)
cv.waitKey(0)





