import cv2
import numpy as np


class Display:
    def __init__(self, *args):
        self.w, self.h = args
        self.frame = self.get_road()
        self.frame_copy = self.frame.copy()

    def get_road(self):
        grass_color, road_color, line_color = (49, 170, 115), (50, ) * 3, (255,) * 3
        boundary, line_length, line_width, centre, start = 100, 50, 5, self.w//2, 50

        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = grass_color
        cv2.rectangle(frame, (boundary, 0), (self.w - boundary, self.h), road_color, -1)
        p = 0
        for i in range(start, self.h + line_length, line_length):
            p += 1
            if p % 2 == 0:
                cv2.rectangle(frame, (centre - line_width, i - line_length), (centre + line_width, i), line_color, -1)

        return frame

    def update(self, *args, **kwargs):
        self.frame = self.frame_copy



if __name__ == "__main__":
    obj = Display(600, 600)
    cv2.imshow("road", obj.frame)
    cv2.waitKey(0)
