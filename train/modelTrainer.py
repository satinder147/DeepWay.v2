import os
import cv2
import sys
import torch
import numpy as np
import argparse
sys.path.append('')
import torch.nn as nn
import torch.optim as optim
from modelArch.unet import Unet
from torchsummary import summary
from dataLoader.dataLoader import load
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
writer=SummaryWriter('runs2/trial1')
def weights_init(m):
    if isinstance(m,nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


net=Unet(3,1)
#net.load_state_dict(torch.load("check/24.pth"))

#net.apply(weights_init)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)

#summary(net,input_size=(3,40,40))
data=load(width=256,height=256)
dataLoader=DataLoader(data,batch_size=4,shuffle=True,num_workers=4)
x=iter(dataLoader)

img,mask=x.next()
grid=torchvision.utils.make_grid(img)
writer.add_image('images',grid,0)
writer.add_graph(net,img)
writer.close()
net.to(device)
#print(img.cpu().numpy().shape)
'''
cv2.imshow("img",np.transpose(img[0].cpu().numpy(),(1,2,0)))
cv2.waitKey(0)
writer.add_graph(net,np.ones((256,256,3)))
writer.close()
'''

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

            #masks=masks.long()
            #print(masks.dtype)
            #masks=masks.squeeze(1)
            #print(masks.shape)
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
            #running_loss += loss.item()
            if(i%20==19):
                print("[%3d] loss:%.10f"%(epoch_num,running_loss/20))
                writer.add_scalar("lr2",running_loss/20,epoch_num*len(dataLoader)+i)
                writer.close()
                running_loss=0.0
                

            #print(img.shape,mask.shape)
            #break
        #l=criterion(outputs,masks)
        torch.save(net.state_dict(),"check/"+str(epoch_num)+".pth")
        
    
if __name__=="__main__":
    #pass
    trainingLoop(epochs=50,lr=1e-4)