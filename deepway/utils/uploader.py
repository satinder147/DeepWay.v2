import os
from shutil import copyfile

path = "dataSet/Segmentation2/mask/"
dest = "masks"
os.mkdir(dest)
files = os.listdir(path)
for i in files:
    num = int(i.split(".")[0])
    if num % 20 == 0:
        copyfile(path+i, dest+"/"+i)

