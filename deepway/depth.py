import os
import cv2
import depthai
import numpy as np
import consts.resource_paths
from hardware.controll_arduino import Arduino
from depthai_helpers.cli_utils import parse_args

global args, cnn_model2
arduino_enabled = False

if arduino_enabled:
    ard = Arduino()

device = depthai.Device('', False)
try:
    args = vars(parse_args())
except:
    os._exit(2)

stream_list = args['streams']
stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in stream_list]    
p = device.create_pipeline(config={
    'streams': stream_list,
    'depth':
        {
            'calibration_file': consts.resource_paths.calib_fpath,
            'padding_factor': 0.3,
            'depth_limit_m': 10.0,  # In meters, for filtering purpose during x,y,z calc
            'confidence_threshold': 0.5,
            # Depth is calculated for bounding boxes with confidence higher than this number
        },
    'ai':
        {
            "blob_file": """/home/satinders/Documents/personal projects/deepway/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.blob""",
            "blob_file_config": "/home/satinders/Documents/personal projects/deepway/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.json"
        },
    # object tracker
    'ot':
        {
            'max_tracklets': 20,  # maximum 20 is supported
            'confidence_threshold': 0.5,  # object is tracked only for detections over this threshold
        },
    'board_config':
        {
            'swap_left_and_right_cameras': args['swap_lr'],
            # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
            'left_fov_deg': args['field_of_view'],  # Same on 1097 and 1098OBC
            'rgb_fov_deg': args['rgb_field_of_view'],
            'left_to_right_distance_cm': args['baseline'],  # Distance between stereo cameras
            'left_to_rgb_distance_cm': args['rgb_baseline'],  # Currently unused
            'store_to_eeprom': args['store_eeprom'],
            'clear_eeprom': args['clear_eeprom'],
            'override_eeprom': args['override_eeprom'],
        },
    'camera':
        {
            'rgb':
                {
                    # 3840x2160, 1920x1080
                    # only UHD/1080p/30 fps supported for now
                    'resolution_h': args['rgb_resolution'],
                    'fps': args['rgb_fps'],
                },
            'mono':
                {
                    # 1280x720, 1280x800, 640x400 (binning enabled)
                    'resolution_h': args['mono_resolution'],
                    'fps': args['mono_fps'],
                },
        },
    'app':
        {
            'sync_video_meta_streams': args['sync_video_meta'],
        },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")
entries_prev = []
from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd
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
            cv2.imshow('previewout', frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packetData
            if args['draw_bb_depth']:
                camera = args['cnn_camera']
                if packet.stream_name == 'disparity':
                    if camera == 'left_right':
                        camera = 'right'
                elif camera != 'rgb':
                    camera = packet.getMetadata().getCameraName()
                # show_nn(nnet_prev["entries_prev"][camera], frame_bgr, labels=labels, config=config, nn2depth=nn2depth)
            cv2.imshow("window_name", frame_bgr)
        elif packet.stream_name == 'jpegout':
            jpg = packetData
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpegout', mat)

        elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
            frame = packetData
            if len(frame.shape) == 2:
                if frame.dtype == np.uint8:  # grayscale
                    pass
                else:  # uint16
                    frame = (65535 // frame).astype(np.uint8)
                    # colorize depth map, comment out code below to obtain grayscale
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                # show_nn(nnet_prev["entries_prev"][camera], frame, labels=labels, config=config, nn2depth=nn2depth)
            cv2.imshow("depth_map", frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device

#  Study streams a bit .
