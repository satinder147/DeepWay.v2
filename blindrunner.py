'''
Things to complete by tonight
1) make a proper blinderRunner.py
2) make pytorch to trt converters
3) runn the system with face detection and stop sign detection tonight
4) run python file on startup
5) start updating the readme add to TODO convert unet to tensorrt
6) look for ways to improve lane prediction
7) if lane prediction happens, use exponential avaerges for dealing with classification errors
8) make a swap file
'''

import cv2
import torch
import time
from torchvision import transforms
from PIL import Image
import numpy as np
#from torchsummary import summary
from modelArch.cnn import Cnn
from modelArch.unet import Unet
from torch2trt import TRTModule
#net=Unet(3,1)



cnn=TRTModule()
cnn.load_state_dict(torch.load("lane_trt.pth"))
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)
trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
cap=cv2.VideoCapture("dataSet/videos/first.mp4")
dic={0:"left",1:"center",2:"right"}


while True:
    with torch.no_grad():
        start=time.time()
        img=cap.read()[1]
        show=img.copy()
        #show=cv2.resize(show,(640,480))
        img=cv2.resize(img,(256,256))
        img=trans(img).unsqueeze(0)
        img=img.to(device)
        label=net(img)
        lane=torch.argmax(label[0].cpu().detach()).numpy()
        end=time.time()
        fps=str(int(1/(end-start))_
        cv2.putText(show,dic[lane.item()]+"  fps: "+fps,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        #cv2.imshow("img",show)
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

