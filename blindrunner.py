from modelArch.unet import Unet
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
net=Unet(3,1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ",device)
net.to(device)

trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
net.load_state_dict(torch.load("checkpoints/5.pth"))
cap=cv2.VideoCapture("dataSet/videos/second.mp4")
while True:

    #img=cv2.imread("dataSet/segmentation1/img/2.jpg",1)
    img=cap.read()[1]
    img=cv2.resize(img,(60,60))
    cv2.imshow("ds",img)
    img=trans(img).unsqueeze(0)
    img=img.type(torch.FloatTensor)
    img=img.to(device)
    out=net(img)

    img=out[0].cpu().detach().numpy()
    img=np.transpose(img,(1,2,0))
    print(img.shape)
    img=img*0.5+0.5
    cv2.imshow("sda",img)
    cv2.waitKey(1)
#img.show()