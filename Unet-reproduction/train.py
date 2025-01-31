import torch
from shapely.coordinates import transform
from torch.utils.data import DataLoader,random_split
from CustomDataset import Carvana
from UnetModel.Model import unet
from torch import optim
from tqdm import tqdm
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from evaluate import PA


if __name__ == '__main__':

    batch_size = 8
    num_epoches = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.000005
    num_workers = 2
    pin_memory = True
    print(f"Now we use {device} to train the model ")
    train_path = "./dataset/train"
    train_mask_path = "./dataset/train_masks"
    image_w = 240
    image_h = 160
    train_percent = 0.8
    trans = A.Compose([A.Resize(height=image_h,width=image_w),A.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    ds = Carvana(train_path,train_mask_path,trans)
    num_train = int(train_percent*len(ds))
    num_val = len(ds)-num_train
    train_ds,valid_ds = random_split(ds,[num_train,num_val])
    train_dl = DataLoader(train_ds,batch_size=batch_size,num_workers=2)
    valid_dl = DataLoader(valid_ds,batch_size=batch_size)
    model = unet(3, 1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(1, num_epoches + 1):

        train_loss = 0
        num = 0
        all = len(train_ds)
        for x, y in train_dl:
            x, y = x.to(device), y.float().to(device)
            y_pred = model(x)
            y = y.unsqueeze(1).float()
            print(y.max())
            print(y_pred)
            optimizer.zero_grad()
            loss = loss_fn(target=y, input=y_pred)
            loss.backward()
            optimizer.step()
            num += batch_size
            train_loss += loss
            print(f"{num / all}", end='\r')

        print(f"in epoch {epoch},the loss is {train_loss}")
        with torch.no_grad():
            accuracy = 0
            count = 0
            for x,y in valid_dl:
                x,y = x.to(device),y.to(device)
                y_pred = model(x)
                y_pred = y_pred.float()
                accuracy += PA(y_pred,y)
                count += 1
        torch.save(model, f"models/unet_model{epoch}")
        print(f"in epoch {epoch},the PA is {accuracy/count}")


