import argparse
import json
import cv2 as cv

from utils.utils_load import VideoReader, GazeReader
from utils.utils_spiral import create_spiral, blank_spiral
from utils.utils_slitscan import scanlines_from_files

parser = argparse.ArgumentParser()
parser.add_argument('--gaze', required=True, type=str)
parser.add_argument('--video', required=True, type=str)
parser.add_argument('--gaze_config', required=True, type=str)
parser.add_argument('--live_preview', action='store_true')
args = parser.parse_args()

with open(args.gaze_config, 'r') as f:
    gaze = GazeReader(args.gaze, f)
    video = VideoReader(args.video)

    with open('configurations/config_spiral.json', 'r') as clock_file:
        spiral_config = json.load(clock_file)
        scanlines = scanlines_from_files(video, gaze, spiral_config['slitscan'])

        spiral = blank_spiral(video.videoCaptureFrameCount, spiral_config)
        spiral = create_spiral(scanlines, spiral, spiral_config, live_preview=args.live_preview)

        cv.imwrite('spiral.png', spiral)
        