from UnetModel.ModelParts import *
from torch import nn
"""n_channels表示的是输入图像的尺寸，RGB图为3，灰度图为1，n_classes为分割的类别数，最后会生成一个（n_classes,H,W）的图像，表示n_classes
张H×W的照片，每张照片表示某种类别的分割图"""
class unet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(unet,self).__init__()
        self.inital_operation = DoubleConv(n_channels,64) #第一次卷积没有下采样操作，从第二次卷积开始，因为源码中是先下采样后卷积
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        if bilinear:
            factor = 2
        else:
            factor = 1
        self.down4 = down(512,1024//factor)
                                                                #尚不明确源码在这块对factor的意义，所以先按照论文上面的模型，使用转置卷积进行上采样
        self.up1 = up(1024,512//factor,bilinear)
        self.up2 = up(512,256//factor,bilinear)
        self.up3 = up(256,128//factor,bilinear)
        self.up4 = up(128,64,bilinear)
        self.out = outConv(64,n_classes)


    def forward(self,x):
        x1 = self.inital_operation(x)  #因为下采样过程中的输出都要保存，与上采样中的结果相加，所以要单独用变量替代
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x,x4) #up中第一个形参要上采样，所以第二个形参为下采样的结果
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        return self.out(x)

