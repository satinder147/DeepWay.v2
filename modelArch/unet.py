import torch
import torch.nn as nn

class Unet(nn.Module):
    '''U-Net Architecture'''

    def __init__(self,inp,out):
        super(Unet,self).__init__()
        self.c1=self.contracting_block(inp,128)
        self.c2=self.contracting_block(128,256)
        self.c3=self.contracting_block(256,512)
        self.maxpool=nn.MaxPool2d(2)
        self.upsample=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.c4=self.contracting_block(256+512,256)
        self.c5=self.contracting_block(128+256,128)
        self.c6=nn.Conv2d(128,out,1)

    def contracting_block(self,inp,out,k=3):
        block =nn.Sequential(
            nn.Conv2d(inp,out,k,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out),
            nn.Conv2d(out,out,k,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out)
        )
        return block


    def forward(self,x):
        #280,280,1
        conv1=self.c1(x) 
        #280,280,128
        x=self.maxpool(conv1)
        #140,140,128
        conv2=self.c2(x)
        #140,140,256
        x=self.maxpool(conv2)
        #70,70,256
        conv3=self.c3(x)
        #70,70,512
        x=self.upsample(conv3)
        #140,140,512
        x=torch.cat([conv2,x],axis=1)
        #140,140,768
        x=self.c4(x)
        #140,140,256
        x=self.upsample(x)
        #280,280,256
        x=torch.cat([conv1,x],axis=1)
        #280,280,384
        x=self.c5(x)
        #280,280,128
        x=self.c6(x)
        #280,280,3
        return x


if __name__=="__main__":
    x=torch.ones(1,3,280,280)
    net=unet(3,3)
    print(net(x).shape)