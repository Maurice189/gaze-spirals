import argparse
from VideoReader import VideoReader
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

kwargs = {'LINE_WIDTH': 2, 'LINE_HEIGHT': 300, 'ANGLE_K': 0.8, 'SAMPLING': 1, 'VERTICAL_CROP': 50, 'VERTICAL_FOCUS': 0.7, 'SPECTOGRAM_HEIGHT': 0}

with open(args.gaze_config, 'r') as f:
    config = json.load(f)
    gaze, special_lines = parse_gaze(args.gaze, config)

    gaze['FRAME'] = gaze['FRAME'].astype(int)
    print(gaze)
    print(special_lines)

    video = VideoReader(args.video)

    #slitscans = create_slitscan(video, gaze, kwargs)

    #cv.imwrite('slitscan.png', slitscans['normal'])
    #cv.imwrite('slitscan_global.png', slitscans['global'])
    #cv.imwrite('slitscan_center.png', slitscans['center'])
    slitscans = {'normal': cv.imread('slitscan.png'), 'global': cv.imread('slitscan_global.png'), 'center': cv.imread('slitscan_center.png')}
    scanlines = scanlines_from_slitscan(slitscans['global'], kwargs)

    spiral = blank_spiral(slitscans['global'], kwargs)
    spiral = create_spiral(scanlines, spiral, kwargs, live_preview=args.live_preview)

    cv.imwrite('spiral.png', spiral)
    