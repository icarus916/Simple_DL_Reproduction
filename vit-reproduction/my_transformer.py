"""
在vit.py中我使用了torch库中封装的transformer模型
本文件中我将自己对其进行实现
"""
from torch import nn
import torch

class Feedforward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super(Feedforward,self).__init__()
        self.process = nn.Sequential(nn.LayerNorm(dim),
                                     nn.Linear(dim,hidden_dim),
                                     nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim,dim),
                                     nn.Dropout(dropout)
                                     )

    def forward(self,x):
        return self.process(x)


class Attention(nn.Module):
    def __init__(self,d_model,heads,dropout):
        super(Attention,self).__init__()
        self.d_model = d_model
        self.heads = heads

        self.k_w = nn.Linear(d_model,d_model*heads)
        self.q_w = nn.Linear(d_model,d_model*heads)
        self.v_w = nn.Linear(d_model,d_model*heads)
        self.norm = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(d_model*heads,d_model)
    def forward(self,x):
        x = self.norm(x)
        batch_num = x.shape[0]
        squ_len = x.shape[1]
        k = self.k_w(x).view(batch_num,squ_len,self.heads,-1).transpose(1,2)
        q = self.q_w(x).view(batch_num,squ_len,self.heads,-1).transpose(1,2)
        v = self.v_w(x).view(batch_num,squ_len,self.heads,-1).transpose(1,2)
        attention = torch.matmul(q,k.transpose(-2,-1))*self.scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        out = torch.matmul(attention,v)
        out = out.transpose(1,2).contiguous().reshape(batch_num,squ_len,-1)
        return self.to_out(out)

if __name__ =='__main__':
    test_tensor = torch.randn(size=(10,1024,128))
    a = Attention(128,8,0.)
    print(a(test_tensor).shape)


