from pupil_labs.realtime_api.simple import Device
import numpy as np

import argparse
from spiral import create_spiral, blank_spiral
from slitscan import scanlines_from_device

parser = argparse.ArgumentParser()
parser.add_argument('--ip', required=True, type=str)
parser.add_argument('--port', required=False, type=str, default='8080')
args = parser.parse_args()

kwargs = {'LINE_WIDTH': 8, 'LINE_HEIGHT': 500, 'ANGLE_K': 0.9, 'SAMPLING': 1, 'VERTICAL_CROP': 50, 'VERTICAL_FOCUS': 0.7, 'SPECTOGRAM_HEIGHT': 0}
#ip = "129.69.207.62"

device = Device(address=args.ip, port=args.port)

print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.serial_number_glasses}")

scanlines = scanlines_from_device(device, kwargs)
spiral = blank_spiral(np.zeros(1000, 1000), kwargs)
create_spiral(scanlines, spiral, kwargs, live_preview=True)