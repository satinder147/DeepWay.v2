import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        print("CNN training")
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(2048, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class LoadCnn(Dataset):
    def __init__(self, **kwargs):

        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.samples = []
        self.dic = {"left": 0, "center": 1, "right": 2}
        self.returnSamples("/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/left",
                           "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/center",
                           "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/lane/right")
        self.transforms_img = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def return_samples(self, *args):

        for path in args:
            img_folder = os.listdir(path)
            for i in tqdm(img_folder):
                p = path.split('/')[-1]
                label = self.dic[p]
                self.samples.append((path + "/" + i, label))

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
        img = np.transpose(img.numpy(), (1, 2, 0))
        # img=img*0.5+0.5
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    sample = torch.ones((1, 3, 32, 32))
    obj = Cnn()
    print(obj.forward(sample).shape)

    # obj = LoadCnn(width=512, height=512)
    # res = obj(5000)
    # obj.plot(res[0])
    # print(res[1])




