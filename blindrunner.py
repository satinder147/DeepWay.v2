from modelArch.unet import Unet
#from unet.unet_model import UNet
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from torchsummary import summary
from modelArch.cnn import Cnn
net=Cnn()
net.load_state_dict(torch.load("lane.pth"))
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)
net.to(device)
summary(net,input_size=(3,64,64))
trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

cap=cv2.VideoCapture("dataSet/videos/first.mp4")
dic={0:"left",1:"center",2:"right"}
while True:
    with torch.no_grad():
        img=cap.read()[1]
        #img=cv2.imread("/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/segmentation1/img/85.jpg",1)
        show=img.copy()
        show=cv2.resize(show,(640,480))
        img=cv2.resize(img,(64,64))
        img=trans(img).unsqueeze(0)
        #img=img.type(torch.FloatTensor)
        img=img.to(device)
        label=net(img)
        lane=torch.argmax(label[0].cpu().detach()).numpy()
        #print(dic[lane.item()])
        cv2.putText(show,dic[lane.item()],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow("img",show)
        cv2.waitKey(1)



        '''
        #print(mask.shape)
        mp=mask[0].cpu().detach().numpy()
        mp=np.transpose(mp,(1,2,0))   #segmentation working
        mp=mp*0.5+0.5
        #mask=mp*0.5+0.5
        #print(mask.shape)
        cv2.imshow("mask",mp)
        img=img.squeeze(0).cpu().detach().numpy()
        img=np.transpose(img,(1,2,0))
        #img=img*0.5+0.5
        cv2.imshow("img",img)
        cv2.waitKey(1)
        #break
'''
