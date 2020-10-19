import cv2
import numpy as np
from scipy import stats
from collections import deque
from pydub import AudioSegment
from pydub.playback import play
from models.unet_keras import Models
from keras.preprocessing.image import img_to_array


def play_audio(name):
    song = AudioSegment.from_mp3(name)
    play(song)


class Lanes:
    """
    This class is responsible for extracting lane lines,
    and then predicting in which direction to move.
    """
    def __init__(self):
        model = Models(256, 256, 3)
        self.model = model.arch3()
        self.model.load_weights("trained_models/lane_lines.MODEL")
        self.deq = deque(maxlen=10)

    @staticmethod
    def lane(lines):
        """
        Args:
            lines: Since hough line transform detections multiple lines for a single line
            so the resulting lane line is the average of all those. Or we can fit a line through all
            those two dimensional points.

        Returns: average lane line slope and intercept.
        """
        xs = []
        ys = []
        for x1, y1, x2, y2 in lines:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
        if len(xs):
            slope, intercept, _, _, _ = stats.linregress(xs, ys)
            return slope, intercept
        return 1e-3, 1e-3

    @staticmethod
    def side(x, y, x1, y1, x2, y2):
        """
        Args:
            x,y:  coordinates of the person
            x1, y1, x2, y2 : starting and ending points of the center, left, right line.
        Returns: Position of the person w.r.t the left, center, right line.
        """
        return (y2 - y1) * (x - x1) - (y - y1) * (x2 - x1)

    def meann(self):
        """
        Returns: because the output from a neural network can subject to noise. To prevent this
        we take average of the last 10 (or I say n) readings.
        """
        s1 = []
        i1 = []
        s2 = []
        i2 = []
        for i in self.deq:
            s1.append(i[0])
            i1.append(i[1])
            s2.append(i[2])
            i2.append(i[3])
        return sum(s1) / len(s1), sum(i1) / len(s1), sum(s2) / len(s1), sum(i2) / len(s1),

    @staticmethod
    def position(d1, d2, d3):
        """
        Returns: the position of the person on the road.
        """
        label = None
        if d1 > 0:
            label = "offroad->left"
        elif d2 > 0:
            label = "left region"
        elif d3 < 0:
            label = "offroad->right"
        elif d2 < 0:
            label = "right region"

        return label

    def get_lines(self, frame):
        """
        Args:
            frame: input image

        Returns: returns the position of the person on the road.

        """
        frame = cv2.resize(frame, (256, 256))
        frame2 = frame.copy()
        frame = cv2.blur(frame, (3, 3))
        frame = img_to_array(frame)
        frame = frame.astype("float") / 255.0
        frame = np.expand_dims(frame, axis=0)
        prediction = self.model.predict(frame)[0]
        mask = prediction * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype("uint8")
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, maxLineGap=100, minLineLength=5)
        slopes = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 != x2:
                        slope = (y2 - y1) / (x2 - x1)
                        slopes.append((slope, [x1, y1, x2, y2]))
        slopes.sort(key=lambda x: x[0])
        left = []
        right = []
        # Sorting the slopes list and then taking mean, those with slopes less than mean and those with greater
        # than mean are the slopes of the two lane lines.
        mean = 0
        for i in range(len(slopes)):
            mean += slopes[i][0]
        mean /= len(slopes)
        for i in range(len(slopes)):
            if slopes[i][0] < mean:
                left.append(slopes[i][1])
            else:
                right.append(slopes[i][1])

        s1, i1 = Lanes.lane(left)
        s2, i2 = Lanes.lane(right)
        self.deq.append((s1, i1, s2, i2))
        s1, i1, s2, i2 = self.meann()
        # calculating the parameters for the center line.
        y1 = 256
        x1 = (y1 - i1) / s1
        x2 = (y1 - i2) / s2
        y2 = 40
        # cv2.line(frame2,(int(x1),y1),(int(x3),y2),(255,0,0),2)
        # cv2.line(frame2,(int(x2),y1),(int(x4),y2),(255,0,0),2)
        # width/2,0 can be taken as the point where the person is w.r.t the complete image.
        person = (128, 256)
        cv2.circle(frame2, person, 6, (0, 255, 0), -6)
        lower = (int((x1 + x2) / 2), 256)
        lower_x = (i2 - i1) / (s1 - s2)
        lower_y = lower_x * s1 + i1
        upper = (int(lower_x), int(lower_y))
        cv2.circle(frame2, lower, 5, (0, 0, 255), -5)
        cv2.circle(frame2, upper, 5, (0, 0, 255), -5)

        d1 = Lanes.side(person[0], person[1], lower[0], lower[1], upper[0], upper[1])
        d2 = Lanes.side(person[0], person[1], x1, y1, upper[0], upper[1])
        d3 = Lanes.side(person[0], person[1], x2, y2, upper[0], upper[1])
        side = Lanes.position(d2, d1, d3)
        # n is the frame number, because I want the servo's to move every 60 frames.
        cv2.putText(frame2, side, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame2, lower, upper, (0, 255, 0), 2)
        cv2.line(frame2, (int(x1), y1), upper, (255, 0, 255), 2)  # right
        cv2.line(frame2, (int(x2), y1), upper, (255, 0, 0), 2)  # left
        cv2.imshow("segmentation", mask)
        cv2.imshow("frame", frame2)
        return side


if __name__ == "__main__":

    cap = cv2.VideoCapture("")
    obj = Lanes()
    while 1:
        ret, frame3 = cap.read()
        obj.get_lines(frame3)
        cv2.waitKey(1)

