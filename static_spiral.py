import argparse
import json
import cv2 as cv

from VideoReader import VideoReader
from lib.gaze_utils.load import parse_gaze
from spiral import create_spiral, blank_spiral
from slitscan import create_slitscan, scanlines_from_slitscan

parser = argparse.ArgumentParser()
parser.add_argument('--gaze', required=True, type=str)
parser.add_argument('--video', required=True, type=str)
parser.add_argument('--gaze_config', required=True, type=str)
parser.add_argument('--live_preview', action='store_true')
args = parser.parse_args()

with open(args.gaze_config, 'r') as f:
    config = json.load(f)
    gaze, special_lines = parse_gaze(args.gaze, config)
    gaze['FRAME'] = gaze['FRAME'].astype(int)
    video = VideoReader(args.video)

    with open('configurations/config_spiral.json', 'r') as clock_file:
        spiral_config = json.load(clock_file)
        slitscan = create_slitscan(video, gaze, spiral_config['slitscan'])
        #cv.imwrite('slitscan.png', slitscans['normal'])
        spiral = blank_spiral(slitscan, spiral_config)

        scanlines = scanlines_from_slitscan(slitscan, spiral_config['slitscan'])
        spiral = create_spiral(scanlines, spiral, spiral_config, live_preview=args.live_preview)
        cv.imwrite('spiral.png', spiral)
        