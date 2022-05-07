import cv2 as cv
import numpy as np

def get_angle(num_sample, angle_k):
    return np.radians(pow(num_sample, angle_k))

def get_radius(angle, height):
    return height*angle / (2*np.pi)

def clock_set_scanline(spiral, line, radius, angle, kwargs):
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    LINE_WIDTH = kwargs['LINE_WIDTH']
    spiralsize = spiral.shape[0]

    xpos = np.cos(angle) * radius
    ypos = np.sin(angle) * radius

    xpos = int(xpos + spiralsize / 2)
    ypos = int(ypos + spiralsize / 2)

    rot_mat = cv.getRotationMatrix2D((LINE_HEIGHT, LINE_HEIGHT), 90-np.degrees(angle), 1)

    dst_patch = spiral[(ypos-LINE_HEIGHT): (ypos+LINE_HEIGHT), (xpos-LINE_HEIGHT): (xpos+LINE_HEIGHT)]
    src_patch = np.zeros((2*LINE_HEIGHT, 2*LINE_HEIGHT, 4), dtype=np.uint8)
    src_patch[LINE_HEIGHT:, LINE_HEIGHT-LINE_WIDTH: (LINE_HEIGHT)+LINE_WIDTH+1, :] = line
    src_patch = cv.warpAffine(src_patch, rot_mat, (2*LINE_HEIGHT, 2*LINE_HEIGHT), flags=cv.INTER_NEAREST)

    src_set = src_patch != 0
    dst_set = dst_patch != 0

    mask = np.logical_xor(src_set, dst_set)
    dst_patch[mask] += src_patch[mask]

def spiral_set_scanline(spiral, line, num_sample, kwargs):
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    LINE_WIDTH = kwargs['LINE_WIDTH']
    ANGLE_K = kwargs['ANGLE_K']
    spiralsize = spiral.shape[0]

    angle = get_angle(num_sample, ANGLE_K)
    radius = get_radius(angle, LINE_HEIGHT)

    xpos = np.cos(angle) * radius
    ypos = np.sin(angle) * radius

    xpos = int(xpos + spiralsize / 2)
    ypos = int(ypos + spiralsize / 2)

    rot_mat = cv.getRotationMatrix2D((LINE_HEIGHT, LINE_HEIGHT), 90-np.degrees(angle), 1)

    dst_patch = spiral[(ypos-LINE_HEIGHT): (ypos+LINE_HEIGHT), (xpos-LINE_HEIGHT): (xpos+LINE_HEIGHT)]
    src_patch = np.zeros((2*LINE_HEIGHT, 2*LINE_HEIGHT, 4), dtype=np.uint8)
    src_patch[LINE_HEIGHT:, LINE_HEIGHT-LINE_WIDTH: (LINE_HEIGHT)+LINE_WIDTH+1, :] = line
    src_patch = cv.warpAffine(src_patch, rot_mat, (2*LINE_HEIGHT, 2*LINE_HEIGHT), flags=cv.INTER_NEAREST)

    src_set = src_patch != 0
    dst_set = dst_patch != 0

    mask = np.logical_xor(src_set, dst_set)
    dst_patch[mask] += src_patch[mask]


def create_spiral(scanlines, spiral, kwargs, live_preview=False):
    num_sample = 0
    for line in scanlines:
        line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
        line[..., 3] = 255
        spiral_set_scanline(spiral, line, num_sample, kwargs)
        num_sample += 1

        if live_preview:
            spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
            cv.imshow('spiral', spiral_downscaled)
            cv.waitKey(1)
    return spiral


def create_clock(scanlines, spiral, kwargs, live_preview=False):
    num_sample = 0
    offset = -1

    for timestamp, line in scanlines:
        if offset == -1:
            offset = timestamp
        timestamp -= offset

        line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
        line[..., 3] = 255

        angle = get_angle(num_sample, angle_k=1)
        radius = get_radius(2*np.pi, kwargs['LINE_HEIGHT'])
        clock_set_scanline(spiral, line, radius, angle, kwargs)

        if timestamp >= 60:
            angle = get_angle(num_sample, angle_k=1)
            radius = get_radius(4*np.pi, kwargs['LINE_HEIGHT'])
            clock_set_scanline(spiral, line, radius, angle, kwargs)
            offset -= 60
        
        num_sample += 1

        if live_preview:
            spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
            cv.imshow('spiral', spiral_downscaled)
            cv.waitKey(1)
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