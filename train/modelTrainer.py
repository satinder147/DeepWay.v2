import sys
import torch
sys.path.append('')
from modelArch.unet import Unet
net=Unet(3,3)
from torchsummary import summary
device=torch.device("cuda")
net.to(device)
summary(net,input_size=(3,280,280))