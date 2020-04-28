import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LoadCnn(Dataset):
    def __init__(self, **kwargs):

        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.samples = []
        self.dic = {"left": 0, "center": 1, "right": 2}
        self.returnSamples("/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/left",
                           "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/center",
                           "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/right")
        self.transforms_img=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def return_samples(self, *args):

        for path in args:
            img_folder = os.listdir(path)
            for i in tqdm(img_folder):
                p = path.split('/')[-1]
                # print(p)
                label = self.dic[p]
                self.samples.append((path+"/"+i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j = self.samples[idx]
        img = cv2.imread(i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.height, self.width))
        img = Image.fromarray(img)
        img = self.transforms_img(img)
        return img, j

    def plot(self, img):
        img = np.transpose(img.numpy(),(1,2,0))
        # img=img*0.5+0.5
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imshow("ds", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    obj = LoadCnn(width=512, height=512)
    res = obj(5000)
    obj.plot(res[0])
    print(res[1])
