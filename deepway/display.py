import cv2
import astar
import numpy as np

name_to_img_name = {'user': 'person.png', 'person': 'pedestrian.png'}


class Display:
    def __init__(self, *args):

        self.w, self.h = args
        self.start = 0
        self.name_img_mask = {}
        self.frame = self.get_road()
        self.frame_copy = self.frame.copy()
        self.load_image_and_preprocess('user')
        self.load_image_and_preprocess("person")

    def load_image_and_preprocess(self, name):
        img = cv2.imread(name_to_img_name[name], -1)
        img = cv2.resize(img, (80, 80))
        mask = img[:, :, 3].astype(np.float)/255.0
        mask = mask.reshape((80, 80, 1))
        img = img[:, :, :3].astype("float")/255.0
        self.name_img_mask[name] = (img, mask)

    def overlay(self, x, y, name):
        if x+40 >= self.w:
            x = self.w - 41
        if x-40 <= 0:
            x = 40
        # print(x, y, name)
        try:
            img, mask = self.name_img_mask[name]
        except KeyError:
            print("This object is not supported in debug mode")
        # print(img.shape, mask.shape)
        bg = self.frame[y-40:y+40, x-40:x+40].astype(np.float)/255.0
        # print(bg.shape, self.frame.shape, x)
        res = img * mask + bg * (1 - mask)
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
        # navigation_map = np.zeros((self.h//2, self.w))
        return frame

    def update(self, label_object_mapping):
        self.frame = self.get_road()
        self.navigation_map = np.zeros((self.h//2, self.w), dtype=np.uint8)
        self.co = self.navigation_map.copy()
        # if isinstance(args, list):
        for label in label_object_mapping:
            for position_x, position_y in label_object_mapping[label]:
                self.overlay(position_x, position_y, label)
        self.start += 1
        self.start %= 100
        # cv2.circle(self.frame, (int(args[0]), 550), 3, (255,)*3, -3)
        x, y = label_object_mapping['user'][0]
        # print(self.navigation_map.shape)

        cv2.circle(self.navigation_map, (324, 125), 10, (255,), -1)
        path = astar.a_star((0, 200), (y-300, x), self.navigation_map)
        path = np.array(path)
        # print(path[[0,1]])
        if path is not False:
            self.co[path[:, 0], path[:, 1]] = 255
            cnt, _ = cv2.findContours(self.co, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cnt[0]
        # print(cnt[0][0][1])

        for i in range(1, len(cnt)):
            x1, y1 = cnt[i][0][0], cnt[i][0][1] + 300
            x2, y2 = cnt[i-1][0][0], cnt[i-1][0][1] + 300
            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    def plot(self):
        cv2.imshow("plot", self.frame)
        cv2.moveWindow("plot", 0, 500)
        cv2.imshow("map", self.navigation_map)
        cv2.moveWindow("map", 800, 500)


if __name__ == "__main__":
    obj = Display(600, 600)
    cv2.imshow("road", obj.frame)
    cv2.waitKey(0)
