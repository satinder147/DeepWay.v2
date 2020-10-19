import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p2', "--path", help="path to input video")
parser.add_argument('-p1', "--spath", help="path where the images should be stored")
parser.add_argument('-n', "--num", help="number from which image label naming starts", type=int)
args = parser.parse_args()
num = args.num
cap = cv2.VideoCapture(args.path)
count = 0
path = args.spath
print(args.num, args.path, args.spath)
ret = True
while ret:
    ret, frame=cap.read()
    count += 1
    if count % 10 == 0:
        cv2.imwrite(path+str(num)+'.jpg', frame)
        print(path+str(num)+'.jpg')
        num += 1
    

