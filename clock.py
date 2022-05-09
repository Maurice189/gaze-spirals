import json
import argparse

from pupil_labs.realtime_api.simple import Device
from utils.utils_clock import blank_clock, create_clock
from utils.utils_linear import scanlines_from_pupil_device


parser = argparse.ArgumentParser()
parser.add_argument('--ip', required=True, type=str)
parser.add_argument('--port', required=False, type=str, default='8080')
args = parser.parse_args()

device = Device(address=args.ip, port=args.port)

print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.serial_number_glasses}")

with open('configurations/config_clock.json', 'r') as clock_file:
    clock_config = json.load(clock_file)
    spiral = blank_clock(clock_config)
    scanlines = scanlines_from_pupil_device(device, clock_config['slitscan'])
    create_clock(scanlines, spiral, clock_config)