import cv2
import os
from keras.models import load_model
from modelArch.unet_keras import Models
from keras.preprocessing.image import img_to_array
import numpy as np
from scipy import stats
from collections import deque

class lanes:
    '''
    Detects lane lines given a binary mask of the road
    '''

    def __init__(self):
        self.deq=deque(maxlen=10)  #next step would be to use the earlier frames for correcting errors in detection
        self.obj=Models(256,256,3)
        self.unet=self.obj.arch3()
        self.unet.load_weights("trained_models/lane_lines.MODEL")
        print("keras unet initialized")
        self.l=None
        self.r=None



    def lane(self,lines):

        xs = []
        ys = []
        for x1, y1, x2, y2 in lines:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
        slope, intercept,_,_,_ = stats.linregress(xs, ys)
        return (slope, intercept)

    def plot_line(self,frame,line,slope,intercept):

        y1,y2=line
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        return (x1,y1,x2,y2,slope,intercept)

    def get_lines(self,frame):
        '''
        args: 256,256,3 frame
        returns: left lines, right line and frame(with lines plotted)
        '''
        frame=cv2.blur(frame,(3,3))
        frame2=frame
        frame=img_to_array(frame)
        frame=frame.astype('float')/255.0
        frame=np.expand_dims(frame,axis=0)
        img=self.unet.predict(frame)[0]
        img=img*255
        img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        t,img2=cv2.threshold(img2,200,255,cv2.THRESH_BINARY)
        img2=img2.astype("uint8")
        img2=cv2.blur(img2,(3,3))
        can=cv2.Canny(img2,150,50)
        can=cv2.dilate(can,None,iterations=3)
        can=cv2.erode(can,None,iterations=3)
        lines=cv2.HoughLinesP(can,1,np.pi/180,50,maxLineGap=50,minLineLength=10)
        if(lines is not None):
            left=[]
            right=[]
            for x in lines:
                
                for x1,y1,x2,y2 in x:
                    if(x2==x1):
                        x2=x2+1
                    s=(y2-y1)/(x2-x1) #slope
                    if(s>=0):
                        left.append([x1,y1,x2,y2])
                    else:
                        right.append([x1,y1,x2,y2])
            self.r=None
            self.l=None

            if(len(left)): 
                sl,il=self.lane(left)
                self.l=self.plot_line(frame2,(50,256),sl,il)
                
            if(len(right)):
                sr,ir=self.lane(right)
                self.r=self.plot_line(frame2,(50,256),sr,ir)

        return self.direction(frame2)
    
    def direction(self,frame2):

        if(self.r is None):
            return "left"
        if(self.l is None):
            return "right"
        if(self.r is not None):
            _,_,_,_,slope1,intercept1=self.r
        if(self.l is not None):
            _,_,_,_,slope2,intercept2=self.l

        x=int((intercept2-intercept1)/(slope1-slope2))
        y=int(slope1*x+intercept1)
        xx1=int((256-intercept1)/slope1)
        xx2=int((256-intercept2)/slope2)

        centerx,centery=128,256
        frame2=cv2.circle(frame2,(centerx,centery),4,(0,255,0),2)
        s, i,_,_,_ = stats.linregress([x,int((xx1+xx2)/2)], [y,256])
        side=s*centerx+i-centery
        frame2=cv2.circle(frame2,(x,y),4,(255,0,0),-2)
        frame2=cv2.line(frame2,(x,y),(int((xx1+xx2)/2),256),(0,255,0),2)
        if(side<0):
            return "left",frame2
        else:
            return "right",frame2


            