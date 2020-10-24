import cv2
import numpy as np
from depth_configs import *
from lane_detection import Lanes
# from hardware.controll_arduino import Arduino
from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd

arduino_enabled = False
try:
    args = vars(parse_args())
except:
    os._exit(2)

lane_lines = Lanes()
if arduino_enabled:
    ard = Arduino()
entries_prev = []


decode_nn = decode_mobilenet_ssd
show_nn = show_mobilenet_ssd

while True:
    neural_net_packets, data_packets = p.get_available_nnet_and_data_packets()
    for packet in data_packets:
        packetData = packet.getData()
        if packet.stream_name not in stream_names:
            continue

        if packetData is None:
            print('Invalid packet data!')
            continue
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for e in entries_prev:
                pt1 = int(e['left'] * img_w), int(e['top'] * img_h)
                pt2 = int(e['right'] * img_w), int(e['bottom'] * img_h)

                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            lane_lines.get_lanes_prediction(frame, True)
            cv2.imshow('preview_out', frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packetData
            if args['draw_bb_depth']:
                camera = args['cnn_camera']
                if packet.stream_name == 'disparity':
                    if camera == 'left_right':
                        camera = 'right'
                elif camera != 'rgb':
                    camera = packet.getMetadata().getCameraName()
            cv2.imshow("window_name", frame_bgr)
        elif packet.stream_name == 'jpeg_out':
            jpg = packetData
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpeg_out', mat)

        elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
            frame = packetData
            if len(frame.shape) == 2:
                if frame.dtype == np.uint8:  # grayscale
                    pass
                else:
                    frame = (65535 // frame).astype(np.uint8)
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            cv2.imshow("depth_map", frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device

