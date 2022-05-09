import argparse
import json
import cv2 as cv
import numpy as np

from utils.utils_linear import scanlines_from_files
from utils.utils_load import VideoReader, GazeReader
from linear import scanlines_from_files
from tqdm import tqdm


def create_blank(width, height, rgb_color=(50, 50, 50)):
    image = np.zeros((width, height, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image    

def linear_slitscan(video, gaze, kwargs):
    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    SAMPLING = kwargs['SAMPLING']
    SPECTOGRAM_HEIGHT = kwargs['SPECTOGRAM_HEIGHT']

    imageWidth = video.videoCaptureFrameCount*(1+LINE_WIDTH*2)
    slitScan = create_blank(LINE_HEIGHT+SPECTOGRAM_HEIGHT, imageWidth)
    curpos = 0

    for scanline in tqdm(scanlines_from_files(video, gaze, kwargs), desc='create_slitscan', unit='Frame', total=video.videoCaptureFrameCount//SAMPLING):
        slitScan[:video.videoHeight, curpos: curpos+(2*LINE_WIDTH+1)] = scanline
        curpos += 2*LINE_WIDTH+1
    return slitScan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaze', required=True, type=str)
    parser.add_argument('--video', required=True, type=str)
    parser.add_argument('--gaze_config', required=True, type=str)
    args = parser.parse_args()

    with open(args.gaze_config, 'r') as f:
        gaze = GazeReader(args.gaze, f)
        video = VideoReader(args.video)

        with open('configurations/config_linear.json', 'r') as clock_file:
            spiral_config = json.load(clock_file)
            slitscan = linear_slitscan(video, gaze, spiral_config['slitscan'])

            line_width = 4000
            line_height = spiral_config['slitscan']['LINE_HEIGHT']

            num_lines = slitscan.shape[1] // line_width
            multiline_slitscan = np.zeros((num_lines*spiral_config['slitscan']['LINE_HEIGHT'], line_width, 3), np.uint8)

            off_x = off_y = 0
            for _ in range(num_lines):
                multiline_slitscan[off_y: off_y+line_height] = slitscan[:, off_x: off_x+line_width]
                off_x += line_width
                off_y += line_height

            cv.imwrite('slitscan.png', slitscan)
            cv.imwrite('multiline_slitscan.png', multiline_slitscan)