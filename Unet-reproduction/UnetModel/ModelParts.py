'''
此处用来存放unet中的doubleConv,down,up,outConv等网络组件，用来组合形成unet网络模型
'''
from matplotlib.scale import scale_factory
from torch import nn
import torch
"""双卷积层，可以理解为是unet中最小的计算单元，down层和up层中都会使用一个双卷积层来提取特征"""
class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel = None):
        super().__init__()
        if mid_channel is None:
            mid_channel = out_channel     #mid_channel 表示第一个卷积的输出通道数和第二个卷积的输入通道数，默认设为输出通道数
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel,mid_channel,kernel_size=3,stride=1,padding=1,bias=False) #论文中的参数即为3 1,但偏置对于模型的影响在论文中尚未体现，目前先根据源码不设置bias偏置
        self.conv2 = nn.Conv2d(mid_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)  #这个inplace允许在原张量上进行运算，对内存的占用会更小
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x)

"""down层，下采样工具，将图像由少通道，大辨识度变为多通道，小辨识度，相当于一个编码器"""
class down(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=None):
        super().__init__()
        self.conv = DoubleConv(in_channel,out_channel,mid_channel)
        self.downSample = nn.MaxPool2d(2)

    def forward(self,x):
        return self.conv(self.downSample(x)) #源码中就是先下采样再卷积

"""上采样层，将多通道小分辨率的图像转化为少通道大分辨率的图像，其中bilinear是指双线性插值法，设为False则使用反卷积"""
class up(nn.Module):
    def __init__(self,in_channel,out_channel,bilinear=True):
        super().__init__()
        if bilinear:
            self.upsample =nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
            self.conv = DoubleConv(in_channel,out_channel,in_channel//2)#中间层的目的应该出于减少过拟合考虑的
        else:
            self.upsample = nn.ConvTranspose2d(in_channel,in_channel//2,2,2)
            self.conv = DoubleConv(in_channel,out_channel)

    def forward(self,x1,x2):         #x1表示在解码器中传递的张量，x2表示在编码器中平行加入的张量
        x1 = self.upsample(x1)
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1,[diffx//2,diffx - diffx//2,diffy,diffy-diffy//2]) #此处的diff表示输入的两个张量x1与x2之间尺寸的偏差
        return self.conv(torch.cat([x2,x1],dim=1)) #表示在第一个维度（通道层）将两种特征相结合（与resnet类似）,其中cat是指其他维度不变，dim维度上张量叠加
                                                        #例如(1,3,5,5)和(1,5,5,5)cat结果为（1，8，5，5）


""""输出层，用于输出分割图"""

class outConv(nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,classes,kernel_size=1)
    def forward(self,x):
        return self.conv(x)










