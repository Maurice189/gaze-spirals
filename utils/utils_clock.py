import numpy as np
import cv2 as cv


def get_angle(num_sample, angle_k):
    return np.radians(pow(num_sample, angle_k))


def get_radius(angle, height):
    return height*angle/(2*np.pi)


def hex2bgr(hex_str):
    r = int(hex_str[1:3], 16)
    g = int(hex_str[3:5], 16)
    b = int(hex_str[5:7], 16)
    return np.array([b, g, r])


def clock_set_scanline(spiral, line, radius, angle, kwargs, shade=False):
    line_height = kwargs['line_height']
    line_width = kwargs['line_width']
    spiralsize = spiral.shape[0]

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

    # TODO: Improve
    if shade:
        #dst_patch[src_set] = (dst_patch[src_set]*1.1).astype(np.uint8)
        dst_patch[src_set] = src_patch[src_set]
    else:
        dst_patch[src_set] = src_patch[src_set]


def create_clock(scanlines, spiral, config):
    offset = -1
    border_width = int(config['border-width'])
    border_color = hex2bgr(config['border-color'])

    for timestamp, line in scanlines:
        if offset == -1:
            offset = timestamp
        timestamp -= offset

        line = cv.cvtColor(line, cv.COLOR_BGR2BGRA)
        line[..., 3] = 255
        line[:border_width, :, :3] = border_color
        line[-border_width:, :, :3] = border_color

        for ring_nr, ring_config in enumerate(config['rings']):
            for t in (timestamp, ):
                y = t % (config['time-unit']**(ring_nr+1))
                y = 360*y/(config['time-unit']**(ring_nr+1))
                y *= ring_config['radial-speed']

                angle = get_angle(y, angle_k=1)
                radius = get_radius(2*np.pi*(ring_nr+1), config['slitscan']['line_height'])

                if t > timestamp:
                    blank_line = np.ones_like(line)
                    clock_set_scanline(spiral, blank_line, radius, angle, config['slitscan'], shade=True)
                else:
                    clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        spiral_downscaled = cv.resize(spiral, (config['window-width'], config['window-height']), interpolation=cv.INTER_CUBIC)
        cv.imshow('spiral', spiral_downscaled)
        cv.waitKey(max(1, config['delay-ms']))


def blank_clock(config):
    line_height = config['slitscan']['line_height']
    line_width = config['slitscan']['line_width']

    border_width = int(config['border-width'])
    border_color = hex2bgr(config['border-color'])

    spiralsize = int(2*get_radius(8*np.pi, line_height))
    spiral = np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)

    for angle in np.arange(0, 360, 0.1):
        line = np.zeros((line_height, 2*line_width+1, 4))
        line[..., :3] = 20
        line[..., 3] = 255

        line[:border_width, :, :3] = border_color
        line[-border_width:, :, :3] = border_color

        angle = get_angle(angle, angle_k=1)

        radius = get_radius(2*np.pi, line_height)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        radius = get_radius(4*np.pi, line_height)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])

        radius = get_radius(6*np.pi, line_height)
        clock_set_scanline(spiral, line, radius, angle, config['slitscan'])
    return spiral

