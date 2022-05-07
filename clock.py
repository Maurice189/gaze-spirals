from pupil_labs.realtime_api.simple import Device
import numpy as np
import json

import argparse
from utils.utils_spiral import blank_clock, create_clock
from linear import scanlines_from_device

parser = argparse.ArgumentParser()
parser.add_argument('--ip', required=True, type=str)
parser.add_argument('--port', required=False, type=str, default='8080')
args = parser.parse_args()

kwargs = {'LINE_WIDTH': 12, 'LINE_HEIGHT': 300, 'ANGLE_K': 0.85, 'SAMPLING': 1, 'VERTICAL_CROP': 50, 'VERTICAL_FOCUS': 0.1, 'SPECTOGRAM_HEIGHT': 0}
device = Device(address=args.ip, port=args.port)

print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.serial_number_glasses}")

with open('configurations/config_clock.json', 'r') as clock_file:
    clock_config = json.load(clock_file)

    #spiral = blank_spiral(slitscans['global'], kwargs)
    spiral = blank_clock(clock_config)
    print(spiral.shape)
    #spiral = create_spiral(scanlines, spiral, kwargs, live_preview=args.live_preview)

    scanlines = scanlines_from_device(device, clock_config['slitscan'])
    create_clock(scanlines, spiral, clock_config)