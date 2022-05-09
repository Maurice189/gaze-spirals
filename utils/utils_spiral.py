import cv2 as cv
import numpy as np


def get_angle(num_sample, angle_k):
    return np.radians(pow(num_sample, angle_k))


def get_radius(angle, height):
    return height*angle / (2*np.pi)


def spiral_set_scanline(spiral, line, num_sample, kwargs):
    line_height = kwargs['slitscan']['line_height']
    line_width = kwargs['slitscan']['line_width']
    ANGLE_K = kwargs['ANGLE_K']
    spiralsize = spiral.shape[0]

    angle = get_angle(num_sample, ANGLE_K)
    radius = get_radius(angle, line_height)

    xpos = np.cos(angle)*radius
    ypos = np.sin(angle)*radius

    xpos = int(xpos+spiralsize/2)
    ypos = int(ypos+spiralsize/2)

    rot_mat = cv.getRotationMatrix2D((line_height, line_height), 90-np.degrees(angle), 1)

    dst_patch = spiral[(ypos-line_height): (ypos+line_height), (xpos-line_height): (xpos+line_height)]
    src_patch = np.zeros((2*line_height, 2*line_height, 4), dtype=np.uint8)
    src_patch[line_height:, line_height-line_width: (line_height)+line_width+1, :] = line
    src_patch = cv.warpAffine(src_patch, rot_mat, (2*line_height, 2*line_height), flags=cv.INTER_NEAREST)

    src_set = src_patch != 0
    dst_patch[src_set] = src_patch[src_set]


def create_spiral(scanlines, spiral, kwargs, live_preview=False):
    num_sample = 0
    for _, line in scanlines:
        line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
        line[..., 3] = 255
        spiral_set_scanline(spiral, line, num_sample, kwargs)
        num_sample += 1

        if live_preview:
            spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
            cv.imshow('spiral', spiral_downscaled)
            cv.waitKey(1)
    return spiral


def blank_spiral(num_frames, kwargs):
    line_height = kwargs['slitscan']['line_height']
    sampling = kwargs['slitscan']['sampling']
    ANGLE_K = kwargs['ANGLE_K']

    steps = num_frames
    max_angle = get_angle(steps//sampling, ANGLE_K) + 2 * np.pi
    spiralsize = int(2*get_radius(max_angle, line_height))
    return np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)