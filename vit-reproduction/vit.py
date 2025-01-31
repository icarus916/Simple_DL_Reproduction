import torch
from torch import nn


class Embd(nn.Module):
    def __init__(self,in_channel=3,patch_size=16,num_patch = 196,embd_dim=3*16*16,dropout=0):
        super(Embd,self).__init__()
        self.patch_embd = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=embd_dim,kernel_size=patch_size,stride=patch_size),
            nn.Flatten(2)
        )                            #编码，相当于将图片进行“图嵌入”
        self.cls_token = nn.Parameter(data=torch.randn(size=(1,1,embd_dim)),requires_grad=True)  #需要一个参数来表示输出结果,这个参数可学习
        self.position = nn.Parameter(data=torch.randn(size=(1,num_patch+1,embd_dim)),requires_grad = True) #位置编码，这里用的是1D编码，可学习
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        cls_token = self.cls_token.expand(x.shape[0],-1,-1) #为了满足批量学习的要求
        x = self.patch_embd(x)
        x = torch.permute(x,(0,2,1))
        x = torch.cat([cls_token,x],dim=1)
        x = x + self.position #这里运用了广播机制所以position不需要expand
        x = self.dropout(x)
        return x



class VIT(nn.Module):
    def __init__(self,in_channel,patch_size,num_patch,embd_dim,dropout,num_head,num_encoders,activation,num_classes):
        super(VIT,self).__init__()
        self.patch_embd = Embd(in_channel,patch_size,num_patch,embd_dim,dropout)
        transformerlayer = nn.TransformerEncoderLayer(d_model=embd_dim,nhead=num_head,dropout=dropout,activation=activation,batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformerlayer,num_layers=num_encoders)
        self.MLP = nn.Sequential(nn.LayerNorm(embd_dim),
                                 nn.Linear(embd_dim,num_classes))

    def forward(self,x):
        x = self.patch_embd(x)
        x = self.transformer(x)
        return self.MLP(x[:,0,:])






if __name__ =='__main__':
    a = torch.randn(size=(10,3,224,224))
    vit = VIT(3,16,14*14,768,0,8,5,'gelu',4)
    print(vit(a).shape)
