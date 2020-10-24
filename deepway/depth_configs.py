import os
import depthai
import consts.resource_paths
from depthai_helpers.cli_utils import parse_args

global args, cnn_model2
arduino_enabled = False
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
            'depth_limit_m': 10.0,
            'confidence_threshold': 0.5,
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