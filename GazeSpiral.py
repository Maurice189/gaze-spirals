import cv2 as cv
import numpy as np
from tqdm import tqdm

SLITSCAN_WIDTH = 3 # How many extra lines next to the scanline should be included
SLITSCAN_HEIGHT = 0 # Set to 0 for the original size of the video, adjust to preferred size
SLITSCAN_CROP = 50 # size of the crop around the point of regard
SLITSCAN_EXP_ANGLE = 0.8
SLITSCAN_SAMPLING = 1
SLITSCAN_GLOBAL_HEIGHT = 300
SLITSCAN_OFF_CENTER_FOCUS = 0.7
SPECTOGRAM_HEIGHT = 1


def create_slitscan(video, gaze):
    imageWidth = video.videoCaptureFrameCount*(1+SLITSCAN_WIDTH*2)
    off_center = int((1-SLITSCAN_OFF_CENTER_FOCUS)*SLITSCAN_HEIGHT)

    slitScan = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)
    slitScan_global = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)
    slitScan_center = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)

    def get_pos(num_sample):
        return SLITSCAN_WIDTH+(SLITSCAN_WIDTH*2+1)*num_sample

    for num_frame in tqdm(range(int(video.videoCaptureFrameCount)), desc='create_slitscan', unit='Frame'):
        entries = gaze[gaze['FRAME'] == num_frame]
        if entries.shape[0] > 0:
            pos_x = int(entries.iloc[0]['GAZE X'])
            pos_y = int(entries.iloc[0]['GAZE Y'])
        else:
            pos_x = pos_y = -1

        img = video.getFrame(num_frame)
        center_x = int(video.videoWidth // 2) 
        curpos = get_pos(num_frame)

        slitScan_center[:video.videoHeight,curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = img[:,center_x-SLITSCAN_WIDTH:center_x+SLITSCAN_WIDTH+1]

        if pos_x >= SLITSCAN_WIDTH and pos_x < video.videoWidth-SLITSCAN_WIDTH and pos_y >= SLITSCAN_CROP and pos_y < video.videoHeight-SLITSCAN_CROP:
            slitScan[:video.videoHeight, curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = img[:, pos_x-SLITSCAN_WIDTH:pos_x+SLITSCAN_WIDTH+1]
            slitScan_global[:video.videoHeight, curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = img[:, pos_x-SLITSCAN_WIDTH:pos_x+SLITSCAN_WIDTH+1]

            scanline = slitScan[:video.videoHeight,curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1]
            scanline[:pos_y-off_center] = scanline[:pos_y-off_center] * 0.7
            scanline[pos_y+off_center:] = scanline[pos_y+off_center: ] * 0.7

            slitScan[:video.videoHeight,curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = scanline
            last_valid = num_frame
                 
        elif num_frame - last_valid < 3:
            lastpos = get_pos(last_valid)
            if lastpos >= SLITSCAN_WIDTH:
                slitScan[0:video.videoHeight,curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = slitScan[0:video.videoHeight,lastpos-SLITSCAN_WIDTH:lastpos+SLITSCAN_WIDTH+1]
                slitScan_global[0:video.videoHeight,curpos-SLITSCAN_WIDTH:curpos+SLITSCAN_WIDTH+1] = slitScan_global[0:video.videoHeight,lastpos-SLITSCAN_WIDTH:lastpos+SLITSCAN_WIDTH+1]

    slitScan = cv.resize(slitScan, (imageWidth,SLITSCAN_GLOBAL_HEIGHT), interpolation = cv.INTER_CUBIC)
    slitScan_global = cv.resize(slitScan_global, (imageWidth,SLITSCAN_GLOBAL_HEIGHT), interpolation = cv.INTER_CUBIC)
    slitScan_center = cv.resize(slitScan_center, (imageWidth,SLITSCAN_GLOBAL_HEIGHT), interpolation = cv.INTER_CUBIC)

    return {'global': slitScan_global, 'center': slitScan_center, 'normal': slitScan}


def create_blank(width, height, rgb_color=(50, 50, 50)):
    """
    This function creates an empty image
    """
    image = np.zeros((width, height, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image    


def spiral_from_stream(next_slitline_fn, live_preview=False):
    def get_angle(num_sample):
        return np.radians(pow(num_sample, SLITSCAN_EXP_ANGLE))

    def get_radius(angle):
        return SLITSCAN_GLOBAL_HEIGHT*angle / (2*np.pi)

    spiralsize = 10000
    spiral = np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)
    num_sample = 0

    while True:
        crop = next_slitline_fn()
        crop = cv.cvtColor(crop, cv.COLOR_RGB2RGBA)
        crop[..., 3] = 255

        angle = get_angle(num_sample)
        radius = get_radius(angle)

        xpos = np.cos(angle) * radius
        ypos = np.sin(angle) * radius

        xpos = int(xpos + spiralsize / 2)
        ypos = int(ypos + spiralsize / 2)

        rot_mat = cv.getRotationMatrix2D((SLITSCAN_GLOBAL_HEIGHT, SLITSCAN_GLOBAL_HEIGHT), 90-np.degrees(angle), 1)

        dst_patch = spiral[(ypos-SLITSCAN_GLOBAL_HEIGHT): (ypos+SLITSCAN_GLOBAL_HEIGHT), (xpos-SLITSCAN_GLOBAL_HEIGHT): (xpos+SLITSCAN_GLOBAL_HEIGHT)]
        src_patch = np.zeros((2*SLITSCAN_GLOBAL_HEIGHT, 2*SLITSCAN_GLOBAL_HEIGHT, 4), dtype=np.uint8)
        src_patch[SLITSCAN_GLOBAL_HEIGHT:, (SLITSCAN_GLOBAL_HEIGHT)-SLITSCAN_WIDTH: (SLITSCAN_GLOBAL_HEIGHT)+SLITSCAN_WIDTH+1, :] = crop
        src_patch = cv.warpAffine(src_patch, rot_mat, (2*SLITSCAN_GLOBAL_HEIGHT, 2*SLITSCAN_GLOBAL_HEIGHT), flags=cv.WARP_FILL_OUTLIERS)

        src_set = src_patch != 0
        dst_set = dst_patch != 0

        mask = np.logical_xor(src_set, dst_set)
        dst_patch[mask] += src_patch[mask]

        if live_preview:
            spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
            cv.imshow('spiral', spiral_downscaled)
            cv.imshow('crop_patch', src_patch)
            cv.waitKey(5)
        num_sample += 1


def create_spiral(slitscan, live_preview=False):
    def get_angle(num_sample):
        return np.radians(pow(num_sample, SLITSCAN_EXP_ANGLE))

    def get_radius(angle):
        return SLITSCAN_GLOBAL_HEIGHT*angle / (2*np.pi)

    steps = int(slitscan.shape[1] / (1+SLITSCAN_WIDTH*2))
    max_angle = get_angle(steps//SLITSCAN_SAMPLING) + 2 * np.pi
    spiralsize = int(2*get_radius(max_angle))
    spiral = np.zeros((spiralsize, spiralsize, 4), dtype=np.uint8)

    with tqdm(total=steps, desc='create_spiral', unit='Frame') as t:
        for num_sample in range(steps // SLITSCAN_SAMPLING):
                pos = SLITSCAN_WIDTH+(SLITSCAN_WIDTH*2+1)*num_sample*SLITSCAN_SAMPLING

                crop = slitscan[:, pos-SLITSCAN_WIDTH: pos+SLITSCAN_WIDTH+1]
                crop = cv.cvtColor(crop, cv.COLOR_RGB2RGBA)
                crop[..., 3] = 255

                angle = get_angle(num_sample)
                radius = get_radius(angle)

                xpos = np.cos(angle) * radius
                ypos = np.sin(angle) * radius

                xpos = int(xpos + spiralsize / 2)
                ypos = int(ypos + spiralsize / 2)

                rot_mat = cv.getRotationMatrix2D((SLITSCAN_GLOBAL_HEIGHT, SLITSCAN_GLOBAL_HEIGHT), 90-np.degrees(angle), 1)

                dst_patch = spiral[(ypos-SLITSCAN_GLOBAL_HEIGHT): (ypos+SLITSCAN_GLOBAL_HEIGHT), (xpos-SLITSCAN_GLOBAL_HEIGHT): (xpos+SLITSCAN_GLOBAL_HEIGHT)]
                src_patch = np.zeros((2*SLITSCAN_GLOBAL_HEIGHT, 2*SLITSCAN_GLOBAL_HEIGHT, 4), dtype=np.uint8)
                src_patch[SLITSCAN_GLOBAL_HEIGHT:, (SLITSCAN_GLOBAL_HEIGHT)-SLITSCAN_WIDTH: (SLITSCAN_GLOBAL_HEIGHT)+SLITSCAN_WIDTH+1, :] = crop
                src_patch = cv.warpAffine(src_patch, rot_mat, (2*SLITSCAN_GLOBAL_HEIGHT, 2*SLITSCAN_GLOBAL_HEIGHT), flags=cv.WARP_FILL_OUTLIERS)

                src_set = src_patch != 0
                dst_set = dst_patch != 0

                mask = np.logical_xor(src_set, dst_set)
                dst_patch[mask] += src_patch[mask]

                if live_preview:
                    spiral_downscaled = cv.resize(spiral, (1000, 1000), interpolation=cv.INTER_CUBIC)
                    cv.imshow('spiral', spiral_downscaled)
                    cv.imshow('crop_patch', src_patch)
                    cv.waitKey(5)
                t.update(SLITSCAN_SAMPLING)
    return spiral


if __name__ == '__main__':
    import argparse
    from VideoReader import VideoReader
    import json

    from VideoReader import VideoReader
    from lib.gaze_utils.load import parse_gaze

    parser = argparse.ArgumentParser()
    parser.add_argument('--gaze', required=True, type=str)
    parser.add_argument('--video', required=True, type=str)
    parser.add_argument('--gaze_config', required=True, type=str)
    args = parser.parse_args()

    with open(args.gaze_config, 'r') as f:
        config = json.load(f)
        gaze, special_lines = parse_gaze(args.gaze, config)

        gaze['FRAME'] = gaze['FRAME'].astype(int)
        print(gaze)
        print(special_lines)

        video = VideoReader(args.video)

        slitscans = create_slitscan(video, gaze)
        cv.imwrite('slitscan.png', slitscans['normal'])
        cv.imwrite('slitscan_global.png', slitscans['global'])
        cv.imwrite('slitscan_center.png', slitscans['center'])

        #slitscans = {'normal': cv.imread('slitscan.png'), 'global': cv.imread('slitscan_global.png'), 'center': cv.imread('slitscan_center.png')}
        spiral = create_spiral(slitscans['global'])

        cv.imwrite('spiral.png', spiral)
        