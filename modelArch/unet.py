import torch
import torch.nn as nn


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
            nn.Conv2d(inp, out, k,padding=1),
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
