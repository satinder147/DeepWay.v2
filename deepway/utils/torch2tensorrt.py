import torch
from torch2trt import torch2trt
from models.cnn import Cnn


def create():
    model_name = "lane_trt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Cnn()
    net = net.to(device)
    x = torch.ones((1, 3, 64, 64)).cuda()
    model_trt = torch2trt(net, [x])
    torch.save(model_trt.state_dict(), model_name+".pth")


if __name__ == "__main__":
    create()


