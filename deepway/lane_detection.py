import cv2
import numpy as np
from scipy import stats
from collections import deque
from models.unet_keras import Models
from keras.preprocessing.image import img_to_array


class Lanes:
    def __init__(self):
        self.model = Models(256, 256, 3)
        self.model = self.model.arch3()
        self.model.load_weights("segmentation.MODEL")
        self.deq = deque(maxlen=10)

    @staticmethod
    def lane(all_lines):
        """
        Use linear regression to draw a line between points.
        Args:
            all_lines (list):
        Returns:
            line_slope
            line_intercept
        """
        xs, ys = [], []
        for line_x1, line_y1, line_x2, line_y2 in all_lines:
            xs.append(line_x1)
            xs.append(line_x2)
            ys.append(line_y1)
            ys.append(line_y2)
        if len(xs):
            line_slope, line_intercept, *_ = stats.linregress(xs, ys)
            return line_slope, line_intercept
        return 1e-3, 1e-3

    @staticmethod
    def side(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
        return (line_y2 - line_y1) * (point_x - line_x1) - (point_y - line_y1) * (line_x2 - line_x1)

    def mean(self):
        """
        Returns average of left and right lane line.
        Args:
            deque:
        Returns:
        """
        slope_1, intercept_1, slope_2, intercept_2 = [], [], [], []
        for element in self.deq:
            slope_1.append(element[0])
            intercept_1.append(element[1])
            slope_2.append(element[2])
            intercept_2.append(element[3])
        num_elements = len(slope_1)
        return sum(slope_1)/num_elements, sum(intercept_1)/num_elements,\
               sum(slope_2)/num_elements, sum(intercept_2)/num_elements

    @staticmethod
    def area(left_line, center_line, right_line):
        label = "none"
        if left_line > 0:
            label = "off-road->left"
        elif center_line > 0:
            label = "left region"
        elif right_line < 0:
            label = "off-road->right"
        elif center_line < 0:
            label = "right region"
        return label

    def get_lanes_prediction(self, frame, debug=False):
        """
        Get lane lines for a frame
        Args:
            frame:
            debug:

        Returns:

        """
        # Get the binary road mask and preprocess it.
        frame = cv2.resize(frame, (256, 256))
        frame_copy = cv2.blur(frame, (3, 3))
        frame = img_to_array(frame_copy)
        frame = frame.astype("float")/255.0
        frame = np.expand_dims(frame, axis=0)
        prediction = self.model.predict(frame)[0]
        prediction = prediction*255
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
        if debug:
            cv2.imshow("binary road mask", prediction)
        prediction = prediction.astype("uint8")
        threshold, prediction = cv2.threshold(prediction, 200, 255, cv2.THRESH_BINARY)
        prediction = cv2.dilate(prediction, None, iterations=2)
        prediction = cv2.erode(prediction, None, iterations=2)
        prediction = cv2.dilate(prediction, None, iterations=2)
        # get lines from the binary mask
        lines = cv2.HoughLinesP(prediction, 1, np.pi/180, 50, maxLineGap=100, minLineLength=5)
        left, right, slopes = [], [], []

        # Separate left and right line based on slope
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 != x2:
                        slope = (y2-y1)/(x2-x1)
                        slopes.append((slope, [x1, y1, x2, y2]))
                        if slope > 0:
                            left.append([x1, y1, x2, y2])
                        else:
                            right.append([x1, y1, x2, y2])

        # Lines with slope less than mean slope are left lines, else right
        slopes.sort(key=lambda x: x[0])
        left2, right2 = [], []
        mean = 0
        for i in range(len(slopes)):
            mean += slopes[i][0]
        mean /= len(slopes)
        # mean=(slopes[0][0]+slopes[len(slopes)-1][0])/2
        for i in range(len(slopes)):
            if slopes[i][0] < mean:
                left2.append(slopes[i][1])
            else:
                right2.append(slopes[i][1])

        # Fit lines through line point using linear regression
        s1, i1 = self.lane(left2)
        s2, i2 = self.lane(right2)
        self.deq.append((s1, i1, s2, i2))
        # Get average of last 10 lane lines
        s1, i1, s2, i2 = self.mean()

        # x coordinate of both the lines at y=256, .i.e. where does the two lane lines intersect the x axis.
        y1 = 256
        x1 = (y1-i1)/s1
        x2 = (y1-i2)/s2

        y2 = 40
        # x3 = (y2-i1)/s1
        # x4 = (y2-i2)/s2
        # cv2.line(frame2,(int(x1),y1),(int(x3),y2),(255,0,0),2)
        # cv2.line(frame2,(int(x2),y1),(int(x4),y2),(255,0,0),2)

        # Position of the person
        person = (128, 256)
        cv2.circle(frame_copy, person, 6, (0, 255, 0), -6)

        # center line coordinates
        lower = (int((x1+x2)/2), 256)
        lower_x = (i2-i1)/(s1-s2)
        lower_y = lower_x*s1+i1
        upper = (int(lower_x), int(lower_y))

        # d=((upper[1]-lower_y)/(upper[0]-lower_x))*person[0]-lower_x+lower_y-person[1]
        cv2.circle(frame_copy, lower, 5, (0, 0, 255), -5)
        cv2.circle(frame_copy, upper, 5, (0, 0, 255), -5)

        # find on which side of the lane lines(all three) the person is currently working
        d1 = self.side(person[0], person[1], lower[0], lower[1], upper[0], upper[1])  # <0-> right else left
        d2 = self.side(person[0], person[1], x1, y1, upper[0], upper[1])  # right <0-> right else left
        d3 = self.side(person[0], person[1], x2, y2, upper[0], upper[1])  # left
        position = self.area(d2, d1, d3)
        # if arduino_enabled:
        #     if position == "off-road->left" and n % 60 == 0:
        #         ard.right()
        #     elif position == "right region" and n % 60 == 0:
        #         ard.left()
        #     elif position == "off-road->right" and n % 60 == 0:
        #         ard.left()
        if debug:
            cv2.putText(frame_copy, position, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame_copy, lower, upper, (0, 255, 0), 2)
            cv2.line(frame_copy, (int(x1), y1), upper, (255, 0, 255), 2)  # right
            cv2.line(frame_copy, (int(x2), y1), upper, (255, 0, 0), 2)  # left
            cv2.imshow("segmentation", prediction)
            cv2.imshow("frame", frame_copy)
        return position


if __name__ == '__main__':
    obj = Lanes()