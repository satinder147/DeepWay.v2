from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch


model_path = "trained_models/mob.pth"
label_path = "trained_models/voc-model-labels.txt"
class detector:
    def __init__(self):

        self.class_names = [name.strip() for name in open(label_path).readlines()]
        num_classes = len(class_names)
        self.net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
        self.net.load(model_path)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=torch.device("cuda:0"))
    
    def return_boxes(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes,labels,prob=self.predictor(img,10,0.4)
        return boxes,labels
        



'''

while True:
    ret, orig_image = cap.read()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''