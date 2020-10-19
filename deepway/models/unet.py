import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Load(Dataset):
    def __init__(self, **kwargs):
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.samples = []
        self.path1 = "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/Segmentation2/img/"
        self.path2 = "/home/satinder/Desktop/deepWay/DeepWay.v2/dataSet/Segmentation2/mask/"
        img_folder = os.listdir(self.path1)

        for i in tqdm(img_folder):
            num = i.split(".")[0]
            self.samples.append((i, num + ".png"))
        self.color = transforms.ColorJitter(brightness=1)
        # self.translate=transforms.RandomAffine(translate=(0.1,0.1))
        self.angle = transforms.RandomAffine(degrees=60)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.transforms_img = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transforms_mask = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j = self.samples[idx]
        img = cv2.imread(self.path1 + i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=cv2.blur(img,(3,3))
        mask = cv2.imread(self.path2 + j, 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(mask, 100, 150)
        mask = cv2.dilate(mask, None, iterations=5)
        img = cv2.resize(img, (self.height, self.width))
        mask = cv2.resize(mask, (self.height, self.width))
        # print(mask.shape)
        seed = np.random.randint(2147483647)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        img = self.transforms_img(img)
        mask = self.transforms_mask(mask)
        # print(img)
        return img, mask

    def plot(self, img):
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = img * 0.5 + 0.5
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", img)
        cv2.waitKey(0)


class Unet(nn.Module):

    def __init__(self, inp, out):
        super(Unet, self).__init__()
        self.c1 = self.contracting_block(inp, 8)
        self.c2 = self.contracting_block(8, 16)
        self.c3 = self.contracting_block(16, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c4 = self.contracting_block(16+32, 16)
        self.c5 = self.contracting_block(8+16, 8)
        self.c6 = nn.Conv2d(8, out, 1)

    def contracting_block(self, inp, out, k=3):
        block = nn.Sequential(
            nn.Conv2d(inp, out, k, padding=1),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out),
            nn.Conv2d(out, out, k, padding=1),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out),
            nn.Conv2d(out, out, k, padding=1),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out)
        )
        return block

    def forward(self, x):
        conv1 = self.c1(x)
        x = self.maxpool(conv1)
        conv2 = self.c2(x)
        x = self.maxpool(conv2)
        conv3 = self.c3(x)
        x = self.upsample(conv3)
        x = torch.cat([conv2, x], axis=1)
        x = self.c4(x)
        x = self.upsample(x)
        x = torch.cat([conv1, x], axis=1)
        x = self.c5(x)
        x = self.c6(x)
        # x=F.sigmoid(x)
        return x


if __name__ == "__main__":
    sample = torch.ones(1, 3, 256, 256)
    net = Unet(3, 1)
    print(net(sample).shape)
    # obj = Load(width=256, height=256)
    # res = obj.__getitem__(7)
    # obj.plot(res[0])
    # obj.plot(res[1])
    # cv2.imshow("img",res[0].cpu().detach().numpy())
