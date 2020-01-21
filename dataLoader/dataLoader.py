import os
import cv2
import t

class load:
    def __init__(self):
        self.images=[]
        self.labels=[]

    def loadData(self,path):
        label=0
        folders=os.listdir(path)
        for i in folders:
            files=os.listdir(path+i)
            for j in files:
                img=cv2.imread(path+i+'/'+j,-1)
                img=cv2.resize(img,(100,100))
                img=img/255.0
                self.images.append(img)
                self.labels.append(label)    
            label=label+1
        return self.images,self.labels


        
        


