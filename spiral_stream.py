from pupil_labs.realtime_api.simple import discover_one_device, Device
from datetime import datetime
import cv2
import numpy as np
import GazeSpiral
from GazeSpiral import SLITSCAN_CROP, SLITSCAN_GLOBAL_HEIGHT, SLITSCAN_HEIGHT, SLITSCAN_WIDTH

ip = "129.69.207.62"
device = Device(address=ip, port="8080")

"""
print(f"Phone IP address: {device.phone_ip}")
print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.serial_number_glasses}")
"""
hd = int(SLITSCAN_GLOBAL_HEIGHT*0.2)

def scanlines():
    while True:
        scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()

        #dt_gaze = datetime.fromtimestamp(gaze_sample.timestamp_unix_seconds)
        #dt_scene = datetime.fromtimestamp(scene_sample.timestamp_unix_seconds)
        #print(f"This gaze sample was recorded at {dt_gaze}")
        #print(f"This scene video was recorded at {dt_scene}")
        #print(f"Temporal difference between both is {abs(gaze_sample.timestamp_unix_seconds - scene_sample.timestamp_unix_seconds) * 1000:.1f} ms")

        img = scene_sample.bgr_pixels

        pos_x = int(gaze_sample.x)
        pos_y = int(gaze_sample.y)

        if pos_x >= SLITSCAN_WIDTH and pos_x < img.shape[1]-SLITSCAN_WIDTH:
            scanline = img[:, pos_x-SLITSCAN_WIDTH: pos_x+SLITSCAN_WIDTH+1].astype(np.uint8)
        else:
            continue

        if pos_y >= hd and pos_y < SLITSCAN_GLOBAL_HEIGHT-hd:
            scanline[:pos_y-hd] = 0.9 * scanline[:pos_y-hd]
            scanline[pos_y+hd: ] = 0.9 * scanline[pos_y+hd:]
        else:
            scanline = scanline * 0.9

        scanline = scanline.astype(np.uint8)
        scanline = cv2.resize(scanline, (2*SLITSCAN_WIDTH+1, GazeSpiral.SLITSCAN_GLOBAL_HEIGHT))

        #cv2.imshow('scene_camera', scanline)
        #cv2.waitKey(1)
        yield scanline


GazeSpiral.spiral_from_stream(scanlines, live_preview=True)
