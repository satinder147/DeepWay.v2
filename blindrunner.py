from modelArch.unet import Unet
#from unet.unet_model import UNet
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from torchsummary import summary
net=Unet(3,1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)
net.to(device)
summary(net,input_size=(3,256,256))
trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
net.load_state_dict(torch.load("checkpoints/5.pth"))
cap=cv2.VideoCapture("dataSet/videos/two.mp4")

while True:
    img=cap.read()[1]
    #img=cv2.imread("/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/segmentation1/img/85.jpg",1)
    img=cv2.resize(img,(256,256))
    img=trans(img).unsqueeze(0)
    img=img.type(torch.FloatTensor)
    img=img.to(device)
    out=net(img)

    out=out[0].cpu().detach().numpy()
    out=np.transpose(out,(1,2,0))
    out=out*0.5+0.5
    cv2.imshow("mask",out)
    img=img.squeeze(0).cpu().detach().numpy()
    img=np.transpose(img,(1,2,0))
    img=img*0.5+0.5
    cv2.imshow("img",img)
    cv2.waitKey(1)

