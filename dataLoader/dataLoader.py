import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms




class load(Dataset):
    def __init__(self,**kwargs):
        self.width=kwargs["width"]
        self.height=kwargs["height"]
        self.samples=[]
        self.path1="/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/segmentation1/img/"
        self.path2="/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/segmentation1/mask/"
        img_folder=os.listdir(self.path1)
        
        for i in tqdm(img_folder):
            num=i.split(".")[0]
            self.samples.append((i,num+".png"))
        self.color=transforms.ColorJitter(brightness = 2,contrast=1,saturation=1,hue=(-0.5,0.5))
        #self.translate=transforms.RandomAffine(translate=(0.1,0.1))
        self.angle=transforms.RandomAffine(degrees=(60))
        self.flip=transforms.RandomHorizontalFlip(p=0.5)
        self.transforms_img=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        self.transforms_mask=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,),(0.5,))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        i,j=self.samples[idx]
        img=cv2.imread(self.path1+i,1)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.path2+j,1)
        img=cv2.resize(img,(self.height,self.width))
        mask=cv2.resize(mask,(self.height,self.width))
        seed=np.random.randint(2147483647)
        img=Image.fromarray(img)
        mask=Image.fromarray(mask)


        random.seed(seed)
        img=self.color(img)
        random.seed(seed)
        #img=self.translate(img)
        random.seed(seed)
        img=self.angle(img)
        random.seed(seed)
        img=self.flip(img)
        random.seed(seed)
        img=self.transforms_img(img)
        
        random.seed(seed)
        #mask=self.translate(mask)
        random.seed(seed)
        mask=self.angle(mask)
        random.seed(seed)
        mask=self.flip(mask)
        random.seed(seed)
        mask=self.transforms_mask(mask)

        return (img,mask)
    
    def plot(self,img):
        img=np.transpose(img.numpy(),(1,2,0))
        img=img*0.5+0.5
        cv2.imshow("ds",img)
        cv2.waitKey(0)


if(__name__=="__main__"):
    obj=load()
    obj.__getitem__(4)
    