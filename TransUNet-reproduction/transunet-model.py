import copy

import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.):
        super(Attention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q = nn.Linear(in_features=d_model,out_features=d_model*num_heads)
        self.k = nn.Linear(in_features=d_model,out_features=d_model*num_heads)
        self.v = nn.Linear(in_features=d_model,out_features=d_model*num_heads)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Linear(in_features=d_model*num_heads,out_features=d_model)
        self.scale = d_model **-0.5
    def forward(self,x):
        batch_size = x.shape[0]
        x = self.norm1(x) #(batch_size,squ,dim)
        q = self.q(x) #(batch_size,squ,hidden_dim*num_heads)
        q = q.view(batch_size,-1,self.num_heads,self.d_model) #(batch_size,squ,num_heads,hidden_dim)
        k = self.k(x)
        k = k.view(batch_size,-1,self.num_heads,self.d_model)
        v = self.v(x)
        v = v.view(batch_size,-1,self.num_heads,self.d_model)
        q,k,v = torch.transpose(q,1,2),torch.transpose(k,1,2),torch.transpose(v,1,2)#(batch_size,num_heads,squ,hidden_dim)
        attention_mat = torch.matmul(q,k.transpose(-1,-2))*self.scale # (batch_size,num_heads,squ,squ)
        attention_mat = self.softmax(attention_mat)
        attention_mat = self.dropout(attention_mat) #(batch_size,num_heads,squ,squ)
        out = torch.matmul(attention_mat,v)
        out = out.transpose(1,2).reshape(batch_size,-1,self.num_heads*self.d_model)

        return self.mlp(out)


class MLP(nn.Module):
    def __init__(self,d_model,hidden_dim,dropout=0.):
        super(MLP,self).__init__()
        self.mlp1 = nn.Linear(d_model,hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim,d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.xavier_uniform_(self.mlp2.weight)
        nn.init.normal_(self.mlp1.bias,std=1e-6)
        nn.init.normal_(self.mlp2.bias,std=1e-6) #经验之谈

    def forward(self,x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        return x

class Embddings(nn.Module):
    def __init__(self,image_size,patch_size,in_channels,hidden_dim = None,dropout=0.):
        super(Embddings,self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        if isinstance(patch_size, tuple) is not True:
            self.patch_size = (patch_size,patch_size)
        if hidden_dim is None:
            hidden_dim = patch_size[0]*patch_size[1]*in_channels
        self.num_patches = (image_size[0]//patch_size[0]) * (image_size[1]//patch_size[1])
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=patch_size,stride=patch_size)
        self.faltten = nn.Flatten(2)
        self.position_embd = nn.Parameter(data=torch.randn(size=(1,self.num_patches,hidden_dim)))
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        x = self.conv(x)
        x = self.faltten(x)
        x = torch.transpose(x,-1,-2) #由conv2d转化后，输出为（Batch，hidden_dim,num_patches）
        x = x + self.position_embd
        x = self.dropout(x)
        return x

"""
论文中的Fig1，照抄就行
"""
class Block(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.):
        super(Block,self).__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model,num_heads,dropout)
        self.mlp = nn.Linear(d_model,d_model)
        nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self,x):
        out = self.layernorm1(x)
        atte = self.attention(out)
        atte = x + atte
        out = self.layernorm2(atte)
        out = self.mlp(out)
        return out + atte


"""
多个注意力层叠加
"""
class Encoder(nn.Module):
    def __init__(self,num_layer,d_model,num_heads,dropout=0.):
        super(Encoder,self).__init__()
        self.layer = nn.ModuleList()
        self.layernorm = nn.LayerNorm(d_model)
        for _ in range(num_layer):
            layer = Block(d_model,num_heads,dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self,x):
        for single_layer in self.layer:
            x = single_layer(x)
        return self.layernorm(x)



class transformer(nn.Module):
    def __init__(self,image_size,patch_size,in_channels,d_model,num_layer,num_heads,dropout=0.,):
        super(transformer,self).__init__()
        self.embd = Embddings(image_size,patch_size,in_channels,d_model,dropout)
        self.encoder = Encoder(num_layer,d_model,num_heads,dropout)

    def forward(self,x):
        x = self.embd(x)
        return self.encoder(x)





if __name__ == '__main__':
    image = torch.randn(size=(10,3,224,224))
    trans = transformer((224,224),(16,16),3,784,10,8)
    print(trans(image).shape)