import argparse
import json
import cv2 as cv
import numpy as np

from utils.utils_linear import scanlines_from_files
from utils.utils_load import VideoReader, GazeReader
from linear import scanlines_from_files
from tqdm import tqdm


def create_blank(width, height, rgb_color=(10, 10, 10)):
    image = np.zeros((width, height, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image    


def create_linear_slitscan(video, gaze, kwargs, target_width=-1, live_preview=True):
    line_width = kwargs['line_width']
    line_height = kwargs['line_height']
    sampling = kwargs['sampling']
    spectogram_height = kwargs['SPECTOGRAM_HEIGHT']

    scan_width = 2*line_width+1
    full_width = video.videoCaptureFrameCount*scan_width

    if target_width == -1:
        target_width = full_width
    else:
        if target_width % scan_width != 0:
            print(f'Warning: The target width of "{target_width} px" is not of a multiple of the scan width "{scan_width} px"!')
        target_width -= target_width % scan_width

    num_lines = np.ceil(full_width/target_width).astype(int)
    canvas = create_blank(num_lines*(line_height+spectogram_height), target_width)
    cursor_x = cursor_y = 0

    for _, scanline in tqdm(scanlines_from_files(video, gaze, kwargs), desc='create_slitscan', unit='Frame', total=video.videoCaptureFrameCount//sampling):
        canvas[cursor_y: cursor_y+line_height, cursor_x: cursor_x+scan_width] = scanline
        if live_preview:
            downscaled = cv.resize(canvas, (1200, 300*num_lines), interpolation=cv.INTER_AREA)
            cv.imshow('live preview (rescaled)', downscaled)
            cv.waitKey(1)

        cursor_x += scan_width
        if cursor_x >= target_width:
            cursor_x = 0
            cursor_y += line_height
    return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaze', required=True, type=str)
    parser.add_argument('--video', required=True, type=str)
    parser.add_argument('--target_width', required=False, type=int, default=-1)
    parser.add_argument('--gaze_config', required=True, type=str)
    parser.add_argument('--live_preview', action='store_true')
    args = parser.parse_args()

    with open(args.gaze_config, 'r') as f:
        gaze = GazeReader(args.gaze, f)
        video = VideoReader(args.video)

        with open('configurations/config_linear.json', 'r') as clock_file:
            config_linear = json.load(clock_file)
            slitscan = create_linear_slitscan(video, gaze, config_linear, target_width=args.target_width, live_preview=args.live_preview)
            cv.imwrite('slitscan.png', slitscan)