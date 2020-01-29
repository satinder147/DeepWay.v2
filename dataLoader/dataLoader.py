import os
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as numpy


class load(Dataset):
    def __init__(self):
        self.samples=[]
        self.path1="/home/satinder/Desktop/deepWay/DeepWay.v2/dataDownloader/img/"
        self.path2="/home/satinder/Desktop/deepWay/DeepWay.v2/dataDownloader/mask/"
        img_folder=os.listdir(self.path1)

        for i in img_folder:
            num=i.split(".")[0]
            self.samples.append((i,num+".png"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        i,j=self.samples[idx]
        img=cv2.imread(self.path1+i,1)
        mask=cv2.imread(self.path2+j,-1)
        mask=mask[:,:,3]
        img=cv2.resize(img,(280,280))
        mask=cv2.resize(mask,(280,280))
        cv2.imshow("mask",mask)
        cv2.imshow("img",img)
        #cv2.waitKey(0)





if(__name__=="__main__"):
    obj=load()
    obj.__getitem__(3)
    