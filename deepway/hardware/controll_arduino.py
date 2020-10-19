import serial
import time


class Arduino:
    def __init__(self):
        self.obj = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
        print("Arduino Connected")

    def left(self):
        self.obj.write(b'1')

    def right(self):
        self.obj.write(b'2')


if __name__ == "__main__":
    obj = Arduino()
    obj.left()
    time.sleep(5)
    obj.right()

    

