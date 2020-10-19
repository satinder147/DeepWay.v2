import cv2
import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

model_path = "trained_models/mob.pth"
label_path = "trained_models/voc-model-labels.txt"


class Detector:
    def __init__(self):
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.num_classes = len(self.class_names)
        self.net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        self.net.load(model_path)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200,
                                                               device=torch.device("cuda:0"))
    
    def return_boxes(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels, prob = self.predictor.predict(img, 10, 0.4)
        return boxes, labels
    
    def label(self, label):
        return self.class_names[label]
        

if __name__ == "__main__":
    obj = Detector()
