import torch
import torchvision

import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset

import time
import torch.nn as nn
import torch.nn.functional as F
from model import model
import cv2
import os

from sklearn.model_selection import train_test_split

output_dir = "/home/divs/Desktop/CSRE/Blur/output"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

gauss = os.listdir('/home/divs/Desktop/CSRE/Blur/Gaussian')

gauss.sort()

sharp =os.listdir('/home/divs/Desktop/CSRE/Blur/sharp')

sharp.sort()

x=[]
y=[]

for i in range(0,len(gauss)):
    x.append(gauss[i])
    y.append(sharp[i])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
tran = transform.Compose([
    transform.ToPILImage(),
    transform.Resize((224,224)),
    transform.ToTensor()
])

class gauss_data(Dataset):

    def __init__(self,blur_path,sharp_path,transforms) -> None:
        self.X = blur_path
        self.Y = sharp_path
        self.dir = '/home/divs/Desktop/CSRE/Blur/'
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        blur_img =cv2.imread(os.path.join(self.dir,"Gaussian",self.X[index]))

        if self.transforms:
            blur_img = self.transforms(blur_img)
        
        if self.Y is not None :
            sharp_img = cv2.imread(os.path.join(self.dir,"sharp",self.Y[index]))
            sharp_img = self.transforms(sharp_img)

            return (blur_img,sharp_img)
        else:
            return blur_img


train_data = gauss_data(x_train,y_train,tran)
test_data = gauss_data(x_test,y_test,tran)

trainloader =DataLoader(train_data,batch_size=4,shuffle=True)
testloader = DataLoader(test_data,batch_size=4,shuffle=False)

model_gauss = model().to(device)

criteria=nn.MSELoss()
optimizer= optim.Adam(model_gauss.parameters(),lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=5,
    factor=0.001,
    verbose=True
)


def train(model,DataLoader):

    model.train()
    running_loss=0

    for i, data in tqdm(enumerate(DataLoader),total=int(len(DataLoader)/DataLoader.batch_size)):

        blur=data[0]
        sharp=data[1]

        blur_image=blur.to(device)
        sharp_image = sharp.to(device)

        optimizer.zero_grad()
        outputs=model(blur_image)
        loss=criteria(outputs,sharp_image)

        loss.backward()

        optimizer.step()

        running_loss+=loss
    
    train_loss = running_loss/len(DataLoader)
    print(f"Train loss:{train_loss}")
    return train_loss

def validate(model,DataLoader):
    model.eval()
    running_loss=0

    with torch.no_grad():
        for i, data in tqdm(enumerate(DataLoader),total=int(len(DataLoader)/DataLoader.batch_size)):

            blur=data[0]
            sharp=data[1]

            blur_image=blur.to(device)
            sharp_image = sharp.to(device)

       
            outputs=model(blur_image)
            loss=criteria(outputs,sharp_image)


            running_loss+=loss.item()

        val_loss=running_loss/len(DataLoader)
        print(f"Val loss:{val_loss}")
        return val_loss


train_loss  = []
val_loss   = []

start=time.time()

for epoch in range(0,100):
    
    train_epoch = train(model_gauss,trainloader)
    val_epoch = validate(model_gauss,testloader)
    train_loss.append(train_epoch)
    val_loss.append(val_epoch)
    scheduler.step(val_epoch)

end=time.time()

print(f"Time taken = {end-start}")

plt.figure()
plt.plot(train_loss,color="red",label='train_loss')
plt.plot(val_loss,color="blue",label='val_loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

torch.save(model.state_dict(),"/home/divs/Desktop/CSRE/Blur/output/model.pth")
