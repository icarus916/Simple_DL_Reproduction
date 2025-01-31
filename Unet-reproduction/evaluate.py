from UnetModel import Model
from torch.utils.data import Dataset,DataLoader
import torch
from torch import Tensor
"""
此文件封装评价分割效果的指标类，源码中使用的是dice_coeff,此文件中我们简化，使用PA(像素准确度)进行评价
PA(target,input) = (target==input).sum() / num_pix
"""

def PA(input: Tensor,target:Tensor):
    input = torch.squeeze(input,1)
    input[input<=0] = 0
    input[input>=0] = 255
    target[target==0] = 0
    target[target==1] = 255
    input = input.type(torch.uint8)
    target = target.type(torch.uint8)
    sum = (input == target).sum().item()
    all = input.shape[0]*input.shape[1]*input.shape[2]
    return sum/all
#
# x = torch.tensor([[[255,255,0],[0,255,255],[255,255,255]],[[255,255,0],[0,255,255],[255,255,255]]])
# y = torch.tensor([[[255,255,0],[255,0,0],[255,0,0]],[[0,255,0],[255,0,0],[255,0,0]]])
#
# print(PA(x,y))
