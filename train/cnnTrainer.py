import sys
sys.path.append('')
import torch
import torch.optim as optim
from modelArch.cnn import Cnn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataLoader.dataloader import load
from torchsummary import summary

data = load(width=64,height=64)
dataloader=DataLoader(data,batch_size=4,shuffle=True,num_workers=4)
net=Cnn()
#net.load_state_dict(torch.load("check4/400.pth"))
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
summary(net,input_size=(3,64,64))
p=100
criterion=nn.CrossEntropyLoss()
opt=optim.Adam(net.parameters(),lr=1e-4)
for ep in range(500):
    running=0.0
    for i,data in enumerate(dataloader):
        opt.zero_grad()
        img,label=data[0].to(device),data[1].to(device)
        output=net(img)

        loss=criterion(output,label)
        running+=loss.item()
        loss.backward()
        opt.step()
        if(i%p==p-1):
            print("[%3d] epochs loss [%.5f]"%(ep,running/p))
            running=0.0
    torch.save(net.state_dict(),"check5/"+str(ep)+".pth")


