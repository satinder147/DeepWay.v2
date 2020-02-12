import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.maxpool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(3,8,3,padding=1)
        self.conv2=nn.Conv2d(8,16,3,padding=1)
        self.conv3=nn.Conv2d(16,32,3,padding=1)
        self.fc1=nn.Linear(8192,100)
        self.fc2=nn.Linear(100,10)
        self.fc3=nn.Linear(10,3)
    
    def forward(self,x):
        x=self.maxpool(F.relu(self.conv1(x)))
        x=self.maxpool(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=x.view(-1,8192)
        #print(x.shape)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x
if __name__=="__main__":
    x=torch.ones((1,3,32,32))
    obj=Cnn()
    print(obj.forward(x).shape)

    
    