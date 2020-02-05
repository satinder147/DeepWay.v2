import os
import sys
import torch
import argparse
sys.path.append('')
import torch.nn as nn
import torch.optim as optim
from modelArch.unet import Unet
from torchsummary import summary
from dataLoader.dataLoader import load
from torch.utils.data import DataLoader

net=Unet(3,1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)
net.to(device)
#summary(net,input_size=(3,40,40))
data=load(width=256,height=256)
dataLoader=DataLoader(data,batch_size=4,shuffle=True,num_workers=4)

def trainingLoop(*args,**kwargs):
    """
    
    """
    if(os.path.isdir("checkpoints")==False):
        os.mkdir("checkpoints")
    epochs=kwargs["epochs"]
    lr=kwargs["lr"]
    criterion=nn.BCEWithLogitsLoss()
    opt=optim.Adam(net.parameters(),lr=lr,weight_decay=1e-8)
    xx=[]
    yy=[]
    for epoch_num in range(1,epochs+1):
        running_loss=0.0
        for i,samples in enumerate(dataLoader):
            imgs,masks=samples[0],samples[1]
            #imgs,masks=imgs.type(torch.FloatTensor),masks.type(torch.long)
            imgs,masks=imgs.to(device),masks.to(device)
            opt.zero_grad()
            outputs=net(imgs)
            #masks=masks.type(torch.float64)
            #print(outputs.shape)
            #print(masks.shape)
            loss=criterion(outputs,masks)
            loss.backward()
            opt.step()
            running_loss += torch.exp(loss).item()
            if(i%20==19):
                print("[%3d] loss:%.3f"%(epoch_num,running_loss/20))
                running_loss=0.0

            #print(img.shape,mask.shape)
            #break
        #l=criterion(outputs,masks)
        torch.save(net.state_dict(),"checkpoints/"+str(epoch_num)+".pth")
        
    
    

trainingLoop(epochs=5,lr=1e-4)