import os
import cv2
import sys
import torch
import argparse
import torchvision
import numpy as np
sys.path.append('')
import torch.nn as nn
import torch.optim as optim
from modelArch.unet import Unet
from torchsummary import summary
from dataLoader.dataLoader import load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser=argparse.ArgumentParser()
parser.add_argument("--epochs",default=50)
parser.add_argument("--batch_size",default=4)
parser.add_argument("--lr",default=0.0001)
args=parser.parse_args()

writer=SummaryWriter('runs2/trial1')

net=None
valid_loader=None
train_loader=None
device=None


def weights_init(m):
    if isinstance(m,nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def init(*args,**kwargs):
    """
    Initiates the training process
    keyword parameters:
    resume:pass checkpoint number from where to resume training
    train_size 
    test_size
    batch_size
    """
    #resume=kwargs["resume"]
    global net,valid_loader,train_loader,device
    resume=None
    train_size=kwargs["train_size"] #400
    test_size=kwargs["test_size"] #26
    batch_size=kwargs["batch_size"]
    net=Unet(3,1)

    if(resume is not None):
        net.load_state_dict(torch.load("checkpoints/"+str(name)+".pth"))
        print("Resuming training from "+str(name)+" checkpoint")
    else:
        net.apply(weights_init)
        
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using ",device)
    #summary(net,input_size=(3,256,256))
    data=load(width=256,height=256)
    print("Data Loaded")
    train,validation=torch.utils.data.random_split(data,[train_size,test_size])
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=4)
    valid_loader=DataLoader(validation,batch_size=batch_size,shuffle=True,num_workers=4)
    #x=iter(dataLoader)
    #img,mask=x.next()
    #grid=torchvision.utils.make_grid(img)
    #writer.add_image('images',grid,0)
    #writer.add_graph(net,img)
    #writer.close()
    net.to(device)

def validation(**kwargs):

    """
    keyword args:
    valid_loader: validation data loader
    """
    global net,valid_loader,train_loader,device
    valid_loader=kwargs["valid_loader"]
    criterion=kwargs["criterion"]
    p=0
    valid_loss=0.0
    with torch.no_grad():
        for no,data in enumerate(valid_loader):
            imgs,masks=data[0].to(device),data[1].to(device)
            outputs=net(imgs)
            v_loss=criterion(outputs,masks)
            valid_loss+=torch.exp(v_loss).item()
            p+=1
    
    return valid_loss/p


def trainingLoop(*args,**kwargs):
    """
    Main training Loop
    keyword parameters:
    epochs:number of epochs
    lr:learning_rate
    
    """
    global net,valid_loader,train_loader,device
    epochs=kwargs["epochs"]
    lr=kwargs["lr"]

    if(os.path.isdir("checkpoints")==False):
        os.mkdir("checkpoints")

    criterion=nn.BCEWithLogitsLoss()
    opt=optim.Adam(net.parameters(),lr=lr,weight_decay=1e-8)
    xx=[]
    yy=[]
    for epoch_num in range(1,epochs+1):
        running_loss=0.0
        for i,samples in enumerate(train_loader):

            imgs,masks=samples[0],samples[1]
            imgs,masks=imgs.to(device),masks.to(device)
            opt.zero_grad()
            outputs=net(imgs)
            loss=criterion(outputs,masks)
            loss.backward()
            opt.step()
            running_loss += torch.exp(loss).item()

            if(i%20==19):
                valid_loss=validation(valid_loader=valid_loader,criterion=criterion)
                writer.add_scalars("first",{'train_loss':torch.tensor(running_loss/20),
                                            'validation_loss':torch.tensor(valid_loss)},epoch_num*len(train_loader)+i)

                writer.close()
                print("Epoch [%3d] iteration [%4d] loss:[%.10f]"%(epoch_num,i,running_loss/20),end="")
                print(" validation_loss:[%.10f]"%(valid_loss))
                running_loss=0.0
        torch.save(net.state_dict(),"checkpoints/"+str(epoch_num)+".pth")
        
    
if __name__=="__main__":
    init(train_size=400,test_size=26,batch_size=int(args.batch_size))
    trainingLoop(epochs=int(args.epochs),lr=int(args.lr))


    