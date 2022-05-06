import cv2 as cv
import numpy as np
from tqdm import tqdm


def create_slitscan(video, gaze, kwargs):
    def create_blank(width, height, rgb_color=(50, 50, 50)):
        image = np.zeros((width, height, 3), np.uint8)
        color = tuple(reversed(rgb_color))
        image[:] = color
        return image    

    def get_pos(num_sample):
        return LINE_WIDTH+(LINE_WIDTH*2+1)*num_sample

    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    VERTICAL_CROP = kwargs['VERTICAL_CROP']
    VERTICAL_FOCUS = kwargs['VERTICAL_FOCUS']
    SPECTOGRAM_HEIGHT = kwargs['SPECTOGRAM_HEIGHT']

    imageWidth = video.videoCaptureFrameCount*(1+LINE_WIDTH*2)
    off_center = int((1-VERTICAL_FOCUS)*LINE_HEIGHT)

    slitScan = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)
    slitScan_global = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)
    slitScan_center = create_blank(video.videoHeight+SPECTOGRAM_HEIGHT,imageWidth)

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

        slitScan_center[:video.videoHeight,curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = img[:,center_x-LINE_WIDTH:center_x+LINE_WIDTH+1]

        if pos_x >= LINE_WIDTH and pos_x < video.videoWidth-LINE_WIDTH and pos_y >= VERTICAL_CROP and pos_y < video.videoHeight-VERTICAL_CROP:
            slitScan[:video.videoHeight, curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = img[:, pos_x-LINE_WIDTH:pos_x+LINE_WIDTH+1]
            slitScan_global[:video.videoHeight, curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = img[:, pos_x-LINE_WIDTH:pos_x+LINE_WIDTH+1]

            scanline = slitScan[:video.videoHeight,curpos-LINE_WIDTH:curpos+LINE_WIDTH+1]
            scanline[:pos_y-off_center] = scanline[:pos_y-off_center] * 0.7
            scanline[pos_y+off_center:] = scanline[pos_y+off_center: ] * 0.7

            slitScan[:video.videoHeight,curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = scanline
            last_valid = num_frame
                 
        elif num_frame - last_valid < 3:
            lastpos = get_pos(last_valid)
            if lastpos >= LINE_WIDTH:
                slitScan[0:video.videoHeight,curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = slitScan[0:video.videoHeight,lastpos-LINE_WIDTH:lastpos+LINE_WIDTH+1]
                slitScan_global[0:video.videoHeight,curpos-LINE_WIDTH:curpos+LINE_WIDTH+1] = slitScan_global[0:video.videoHeight,lastpos-LINE_WIDTH:lastpos+LINE_WIDTH+1]

    slitScan = cv.resize(slitScan, (imageWidth,LINE_HEIGHT), interpolation = cv.INTER_CUBIC)
    slitScan_global = cv.resize(slitScan_global, (imageWidth,LINE_HEIGHT), interpolation = cv.INTER_CUBIC)
    slitScan_center = cv.resize(slitScan_center, (imageWidth,LINE_HEIGHT), interpolation = cv.INTER_CUBIC)
    return {'global': slitScan_global, 'center': slitScan_center, 'normal': slitScan}



def scanlines_from_slitscan(slitscan, kwargs):
    LINE_WIDTH = kwargs['LINE_WIDTH']
    SAMPLING = kwargs['SAMPLING']
    steps = int(slitscan.shape[1] / (1+LINE_WIDTH*2))

    with tqdm(total=steps, desc='create_spiral', unit='Frame') as t:
        for num_sample in range(steps // SAMPLING):
                pos = LINE_WIDTH+(LINE_WIDTH*2+1)*num_sample*SAMPLING
                line = slitscan[:, pos-LINE_WIDTH: pos+LINE_WIDTH+1]
                line = cv.cvtColor(line, cv.COLOR_RGB2RGBA)
                line[..., 3] = 255

                t.update(SAMPLING)
                yield line


def scanlines_from_device(device, kwargs):
    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    VERTICAL_FOCUS = kwargs['VERTICAL_FOCUS']
    #SAMPLING = kwargs['SAMPLING']
    off_center = int((1-VERTICAL_FOCUS)*LINE_HEIGHT)

    while True:
        scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
        img = scene_sample.bgr_pixels

        pos_x = int(gaze_sample.x)
        pos_y = int(gaze_sample.y)

        if pos_x >= LINE_WIDTH and pos_x < img.shape[1]-LINE_WIDTH:
            scanline = img[:, pos_x-LINE_WIDTH: pos_x+LINE_WIDTH+1].astype(np.uint8)
        else:
            continue

        if pos_y >= off_center and pos_y < LINE_HEIGHT-off_center:
            scanline[:pos_y-off_center] = 0.9 * scanline[:pos_y-off_center]
            scanline[pos_y+off_center: ] = 0.9 * scanline[pos_y+off_center:]
        else:
            scanline = scanline * 0.9

        scanline = scanline.astype(np.uint8)
        scanline = cv.resize(scanline, (2*LINE_WIDTH+1, LINE_HEIGHT), interpolation=cv.INTER_CUBIC)

        #cv2.imshow('scene_camera', scanline)
        #cv2.waitKey(1)
        yield scanline
