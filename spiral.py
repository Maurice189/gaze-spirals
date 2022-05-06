import cv2 as cv
import numpy as np


def get_angle(num_sample, angle_k):
    return np.radians(pow(num_sample, angle_k))

def get_radius(angle, height):
    return height*angle / (2*np.pi)

def set_scanline(spiral, line, num_sample, kwargs):
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    ANGLE_K = kwargs['ANGLE_K']

    spiralsize = spiral.shape[0]
    height = line.shape[0]
    slitscan_width = (line.shape[1] - 1) // 2

    angle = get_angle(num_sample, ANGLE_K)
    radius = get_radius(angle, LINE_HEIGHT)

    xpos = np.cos(angle) * radius
    ypos = np.sin(angle) * radius

    xpos = int(xpos + spiralsize / 2)
    ypos = int(ypos + spiralsize / 2)

    rot_mat = cv.getRotationMatrix2D((height, height), 90-np.degrees(angle), 1)

    dst_patch = spiral[(ypos-height): (ypos+height), (xpos-height): (xpos+height)]
    src_patch = np.zeros((2*height, 2*height, 4), dtype=np.uint8)
    src_patch[height:, height-slitscan_width: (height)+slitscan_width+1, :] = line
    src_patch = cv.warpAffine(src_patch, rot_mat, (2*height, 2*height), flags=cv.WARP_FILL_OUTLIERS)

    src_set = src_patch != 0
    dst_set = dst_patch != 0

    mask = np.logical_xor(src_set, dst_set)
    dst_patch[mask] += src_patch[mask]


def create_spiral(scanlines, spiral, kwargs, live_preview=False):
    num_sample = 0
    for line in scanlines:
        line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
        line[..., 3] = 255
        set_scanline(spiral, line, num_sample, kwargs)
        num_sample += 1

        if live_preview:
            spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
            cv.imshow('spiral', spiral_downscaled)
            cv.waitKey(5)
    return spiral

def blank_spiral(slitscan, kwargs):
    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    SAMPLING = kwargs['SAMPLING']
    ANGLE_K = kwargs['ANGLE_K']

    steps = int(slitscan.shape[1] / (1+LINE_WIDTH*2))
    max_angle = get_angle(steps//SAMPLING, ANGLE_K) + 2 * np.pi
    spiralsize = int(2*get_radius(max_angle, LINE_HEIGHT))
    return np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)