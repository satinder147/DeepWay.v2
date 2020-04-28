import cv2
from lane_detection import Lanes
# from hardware.controll_arduino import Arduino
# from pedestrian import Detector

# person = Detector()
obj = Lanes()
cap = cv2.VideoCapture("dataSet/videos/two.mp4")
ret = True
n = 0
while ret:
    n += 1
    ret, frame = cap.read()
    frame2 = frame.copy()
    side = obj.get_lines(frame)
    frame2 = cv2.resize(frame2, (320, 240))

    # boxes, labels = person.return_boxes(frame2)
    # for i in range(boxes.size(0)):
    #     x1, y1, x2, y2 = boxes[i]
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     label = person.label(labels[i])
    #     cv2.rectangle(frame, (x1 + 2, y1 + 2), (x1 + 120, y1 + 30), (255, 255, 255), -2)
    #     cv2.putText(frame, label, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    if side == "offroad->left" and n % 60 == 0:
        pass  # ard.right()
    elif side == "right region" and n % 60 == 0:
        pass  # ard.left()
    elif side == "offroad->right" and n % 60 == 0:
        pass  # ard.left()
    cv2.imshow("pedestrian", frame2)
    if cv2.waitKey(1) and 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
