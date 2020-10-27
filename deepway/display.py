import cv2
import numpy as np


class Display:
    def __init__(self, *args):
        self.w, self.h = args
        self.start = 0
        self.frame = self.get_road()
        self.frame_copy = self.frame.copy()
        self.person = cv2.imread("person.png", -1)
        self.person = cv2.resize(self.person, (80, 80))
        self.person_mask = self.person[:, :, 3].astype(np.float)/255.0
        self.person_mask_inv = cv2.bitwise_not(self.person_mask).astype(np.float)/255.0
        self.person_mask = self.person_mask.reshape((80, 80, 1))
        self.person = self.person[:, :, :3].astype("float")/255.0

    def overlay(self, x, y):
        if x >= self.w:
            x = self.w - 41
        if x <= 0:
            x = 40
        bg = self.frame[y-40:y+40, x-40:x+40].astype(np.float)/255.0
        res = self.person * self.person_mask + bg * (1 - self.person_mask)
        res = (res * 255).astype(np.uint8)
        self.frame[y-40:y+40, x-40:x+40] = res

    def get_road(self):
        grass_color, road_color, line_color = (49, 170, 115), (50, ) * 3, (255,) * 3
        boundary, line_length, line_width, centre, _ = 100, 50, 5, self.w//2, 50

        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = grass_color
        cv2.rectangle(frame, (boundary, 0), (self.w - boundary, self.h), road_color, -1)
        p = 0
        for i in range(self.start, self.h + line_length, line_length):
            p += 1
            if p % 2 == 0:
                cv2.rectangle(frame, (centre - line_width, i - line_length), (centre + line_width, i), line_color, -1)

        return frame

    def update(self, *args, **kwargs):
        self.frame = self.get_road()
        self.overlay(int(args[0]), 550)
        self.start += 1
        self.start %= 100
        # cv2.circle(self.frame, (int(args[0]), 550), 3, (255,)*3, -3)

    def plot(self):
        cv2.imshow("plot", self.frame)
        cv2.moveWindow("plot", 0, 500)


if __name__ == "__main__":
    obj = Display(600, 600)
    cv2.imshow("road", obj.frame)
    cv2.waitKey(0)
