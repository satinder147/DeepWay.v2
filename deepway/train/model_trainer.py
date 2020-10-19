import os
import sys
import math
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

from models.unet import Load
from models.cnn import Cnn, LoadCnn

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=200)
parser.add_argument("--batch_size", default=20)
parser.add_argument("--lr", default=0.0002)
parser.add_argument("--model", default="unet", help="unet/cnn")
args = parser.parse_args()

print("number of epochs: ", args.epochs)
print("batch size: ", args.batch_size)
print("learning rate: ", args.lr)
print("training ", args.model)

writer = SummaryWriter('runs/trial1')
num_iter = 10
net = None
valid_loader = None
train_loader = None
device = None
data = None
model = None


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def init(**kwargs):

    """
    Initiates the training process
    keyword parameters:
    train_percent:[0,1]
    resume:pass checkpoint number from where to resume training
    batch_size
    """

    global net, valid_loader, train_loader, device, data
    resume = None
    train_percent = kwargs["train_percent"]
    batch_size = kwargs["batch_size"]
    width = kwargs["width"]
    height = kwargs["width"]
    if model == "unet":
        net = smp.Unet('mobilenet_v2', encoder_weights='imagenet')
        data = Load(width=width, height=height)

    elif model == "cnn":
        net = Cnn()
        data = LoadCnn(width=width,height=height)

    if resume is not None:
        net.load_state_dict(torch.load("checkpoints/"+str(resume)+".pth"))
        print("Resuming training from "+str(resume)+" checkpoint")
    else:
        pass
        # net.apply(weights_init)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # summary(net,input_size=(3,256,256))
    size = len(data)
    print("Data Loaded")
    train_size = math.floor(train_percent*size)
    test_size = size-train_size
    print("Total size of the dataset: ", size)
    print("Train data size: ", train_size)
    print("Test data size: ", test_size)
    print("using {} for training".format(device))
    train, validation = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)
    # x=iter(dataLoader)
    # img,mask=x.next()
    # grid=torchvision.utils.make_grid(img)
    # writer.add_image('images',grid,0)
    # writer.add_graph(net,img)
    # writer.close()
    net.to(device)


def validation(**kwargs):

    """
    keyword args:
    valid_loader: validation data loader
    """
    global net, valid_loader, train_loader, device
    valid_loader = kwargs["valid_loader"]
    criterion = kwargs["criterion"]
    # model=kwargs["model"]
    p = 0
    valid_loss = 0.0
    with torch.no_grad():
        for no, data in enumerate(valid_loader):
            imgs, masks = data[0].to(device), data[1].to(device)
            num_img = imgs.shape[0]
            outputs = net(imgs)
            v_loss = calc_loss(outputs, masks)
            if model == 'unet':
                valid_loss += (torch.exp(v_loss).item())*(args.batch_size/num_img)
            elif model == 'cnn':
                valid_loss += v_loss.item()
            p += 1
    
    return valid_loss/p


def training_loop(**kwargs):
    """
    Main training Loop
    keyword parameters:
    epochs:number of epochs
    lr:learning_rate
    
    """
    print("Training Loop")
    global net, valid_loader, train_loader, device
    epochs = kwargs["epochs"]
    lr = kwargs["lr"]
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    criterion = None
    if model == "unet":
        criterion = nn.BCEWithLogitsLoss()
    elif model == "cnn":
        criterion = nn.CrossEntropyLoss()
    # opt=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)
    opt = optim.Adam(net.parameters(),lr=lr,weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    # sch=torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
    # sch=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
    xx = []
    yy = []
    for epoch_num in range(1, epochs+1):
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, samples in enumerate(train_loader):
            imgs, masks=samples[0], samples[1]
            num_imgs = imgs.shape[0]
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            outputs = net(imgs)
            loss = calc_loss(outputs, masks)  # criterion(outputs,masks)
            loss.backward()
            opt.step()
            _, predicted = torch.max(outputs.data, 1)
            total_train += masks.nelement()
            correct_train += predicted.eq(masks.data).sum().item()
            running_loss += (torch.exp(loss).item()) * (args.batch_size/num_imgs)
            valid_loss = 0
            if i % num_iter == 0 and i != 0:
                valid_loss=validation(valid_loader=valid_loader, criterion=criterion)
                writer.add_scalars("first", {'train_loss': torch.tensor(running_loss/num_iter),
                                             'validation_loss': torch.tensor(valid_loss)},
                                   epoch_num * len(train_loader) + i)

                writer.close()
                print("Epoch [%3d] iteration [%4d] loss:[%.10f]" %
                      (epoch_num, i, running_loss/num_iter), end="")
                print(" validation_loss:[%.10f]" % valid_loss, end="")
                print("acc [%3d] current lr [%.10f]" % (correct_train/total_train,
                                                        opt.param_groups[0]['lr']))
                running_loss = 0.0
                correct_train = 0
                total_train = 0

            if i == 10:
                sch.step(valid_loss)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        if epoch_num % 2 == 0:
            torch.save(net.state_dict(), "checkpoints/"+str(epoch_num)+".pth")

    
if __name__ == "__main__":
    model = args.model
    init(batch_size=int(args.batch_size), train_percent=0.9, width=224, height=448)
    training_loop(epochs=int(args.epochs), lr=float(args.lr))

