from time import time
import cv2 as cv
import numpy as np

def hex2bgr(hex_str):
    r = int(hex_str[1:3], 16)
    g = int(hex_str[3:5], 16)
    b = int(hex_str[5:7], 16)
    return np.array([b, g, r])

def get_angle(num_sample, angle_k):
    return np.radians(pow(num_sample, angle_k))

def get_radius(angle, height):
    return height*angle / (2*np.pi)

def clock_set_scanline(spiral, line, radius, angle, kwargs, shade=False):
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
    if shade:
        dst_patch[src_set] = (dst_patch[src_set]*1.5).astype(np.uint8)
    else:
        dst_patch[src_set] = src_patch[src_set]

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


def create_clock(scanlines, spiral, config):
    offset = -1
    border_width = int(config['border-width'])
    border_color = hex2bgr(config['border-color'])

    for timestamp, line in scanlines:
        if offset == -1:
            offset = timestamp
        timestamp -= offset

        line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
        line[..., 3] = 255

        line[:border_width, :, :3] = border_color
        line[-border_width:, :, :3] = border_color

        for ring_nr, ring_config in enumerate(config['rings']):
            ring_nr += 1

            for t in (timestamp, timestamp+0.05):
                y = t % (config['time-unit']**ring_nr)
                y = 360*y/(config['time-unit']**ring_nr)
                y *= ring_config['radial-speed']

                angle = get_angle(y, angle_k=1)
                radius = get_radius(2*np.pi*ring_nr, config['slitscan']['LINE_HEIGHT'])

                if t > timestamp:
                    blank_line = np.ones_like(line)
                    clock_set_scanline(spiral, blank_line, radius, angle, config['slitscan'], shade=True)
                else:
                    clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        spiral_downscaled = cv.resize(spiral, (config['window-width'], config['window-height']), interpolation=cv.INTER_CUBIC)
        cv.imshow('spiral', spiral_downscaled)
        cv.waitKey(max(1, config['delay-ms']))


def blank_clock(config):
    LINE_HEIGHT = config['slitscan']['LINE_HEIGHT']
    LINE_WIDTH = config['slitscan']['LINE_WIDTH']

    border_width = int(config['border-width'])
    border_color = hex2bgr(config['border-color'])

    spiralsize = int(2*get_radius(8*np.pi, LINE_HEIGHT))
    spiral = np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)

    for angle in range(360):
        line = np.zeros((LINE_HEIGHT, 2*LINE_WIDTH+1, 4))
        line[..., :3] = 20
        line[..., 3] = 255

        line[:border_width, :, :3] = border_color
        line[-border_width:, :, :3] = border_color

        angle = get_angle(angle, angle_k=1)

        radius = get_radius(2*np.pi, LINE_HEIGHT)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        radius = get_radius(4*np.pi, LINE_HEIGHT)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        radius = get_radius(6*np.pi, LINE_HEIGHT)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])
        
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