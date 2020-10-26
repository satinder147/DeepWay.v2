import os
import cv2
import numpy as np
import depthai
import json
from lane_detection import Lanes
base = "/home/satinders/Documents/personal projects/deepway/depthai/resources/nn"


class DepthAi:
    def __init__(self):
        self.lanes = Lanes()
        self.device = depthai.Device('', False)
        json_path = os.path.join(base, "mobilenet-ssd/mobilenet-ssd_depth.json")
        config = {
            "streams": ["metaout", "previewout"],
            "ai": {
                "calc_dist_to_bb": True,
                "blob_file": os.path.join(base, "mobilenet-ssd/mobilenet-ssd.blob"),
                "blob_file_config": json_path
            }
        }
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        self.labels = list(json_file['mappings']['labels'])
        self.p = self.device.create_pipeline(config=config)
        self.entries = []
        # print(self.p)
        print(config)

    def run(self):
        while 1:
            net_packets, data_packets = self.p.get_available_nnet_and_data_packets()
            for net_packet in net_packets:
                self.entries = []
                for e in net_packet.entries():

                    if e[0]['id'] == -1 or e[0]['confidence'] == 0.0:
                        break
                    if e[0]['confidence'] > 0.5:
                        self.entries.append(e[0])

            for packet in data_packets:
                if packet.stream_name == "previewout":
                    data = packet.getData()
                    blue, green, red = data[0, :, :], data[1, :, :], data[2, :, :]
                    frame = cv2.merge([blue, green, red])
                    img_h, img_w = frame.shape[:2]

                    results = []
                    for e in self.entries:
                        pt1 = int(e['left'] * img_w), int(e['top'] * img_h)
                        pt2 = int(e['right'] * img_w), int(e['bottom'] * img_h)
                        label = self.labels[int(e['label'])]
                        results.append((pt1, pt2, label, e['distance_x'], e['distance_y'], e['distance_z']))
                    yield frame, results

    def __del__(self):
        del self.p
        del self.device


if __name__ == '__main__':
    obj = DepthAi()
    for frame, results in obj.run():
        for pt1, pt2, label, dist_x, dist_y, dist_z in results:
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            print(dist_x, dist_y, dist_z)
            cv2.putText(frame, str(label), pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    del obj
