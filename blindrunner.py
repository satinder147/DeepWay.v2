import os
import cv2
import torch
import time
import numpy as np
from PIL import Image
from modelArch.cnn import Cnn
#from modelArch.unet import Unet  isko baad me bhi comment karke rakhna
from torchvision import transforms
from pedestrian import detector
from hardware.controll_arduino import Arduino
from lane_detection import lanes

#from torchsummary import summary uncomment for taking a look at the model architecture
#from torch2trt import TRTModule  I have been able to run tensorrt model, but don't know why the accuracy is extreamly low.
import cv2
from lane_detection import lanes
from scipy import stats
unet=None
cnn=None
device=None
trans=None
cap=None
cap1=None
dic=None
n=None
seconds=None
person=None
obj=None

def init():
    global unet,cnn,device,trans,cap,dic,seconds,n,person,cap1,obj
    #unet=Unet(3,1)
    cnn=Cnn()
    #ard=Arduino()
    cnn.load_state_dict(torch.load("trained_models/lane.pth"))
    print("lane detection model loaded")
    #unet.load_state_dict(torch.load("trained_models/segmentation.pth"))
    print("road segmentation model loaded")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    #unet.to(device)
    print("using ",device)
    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    cap=cv2.VideoCapture("dataSet/videos/two.mp4")
    dic={0:"left",1:"center",2:"right"}
    n=0
    seconds=3*30
    cap1=cv2.VideoCapture(0)
    person=detector()
    obj=lanes()

def inference():
    global unet,cnn,device,trans,cap,dic,seconds,n,detector,cap1,obj
    while True:
        with torch.no_grad():
            start=time.time()
            n=n+1
            frame=cap1.read()[1]
            frame=cv2.resize(frame,(320,240))
            img=cap.read()[1]
            lane=img.copy()
            segmentation=img.copy()
            show=img.copy()
            lane=cv2.resize(lane,(64,64))
            segmentation=cv2.resize(segmentation,(256,256))
            obj.get_lines(segmentation)
            #frame2=cv2.putText(segmentation,label_unet,(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            boxes,labels=person.return_boxes(frame)
            for i in range(boxes.size(0)):
                x1,y1,x2,y2=boxes[i]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                label=person.label(labels[i])
                
                cv2.rectangle(frame,(x1+2,y1+2),(x1+120,y1+30),(255,255,255),-2)
                cv2.putText(frame,label,(x1+10,y1+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            show=cv2.resize(show,(640,480))

            lane=trans(lane).unsqueeze(0)
            #segmentation=trans(segmentation).unsqueeze(0)

            lane=lane.to(device)
            #segmentation=segmentation.to(device)

            label=cnn(lane)
            lane=torch.argmax(label[0].cpu().detach()).numpy()
            lane=dic[lane.item()]

            # road=unet(segmentation)                             #commenting because pytorch unet 
            # road=road[0].cpu().detach().numpy()                 #performs really poor
            # road=np.transpose(road,(1,2,0))                  
            # road=road*0.5+0.5

            if((lane=="right" or lane=="center") and n%(seconds)==0):
                #ard.left()
                pass

            
            end=time.time()
            fps=str(int(1/(end-start)))
            cv2.putText(show,lane+"  fps: "+fps,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("img",show)
            cv2.imshow("lane lines",frame2)
            cv2.imshow("frame",frame)

            cv2.waitKey(1)



if(__name__=="__main__"):
    init()
    inference()
    trial()
