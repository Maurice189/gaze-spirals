import numpy as np
import cv2 as cv


def transform_frame(img, new_height):
    height, width = img.shape[:2]
    aspect = width / height
    videoWidth = int(new_height*aspect)
    videoHeight = new_height
    return cv.resize(img, (videoWidth, videoHeight), interpolation=cv.INTER_CUBIC)


def transform_gaze(gaze, new_size, old_size):
    pos_x, pos_y = gaze
    new_width, new_height = new_size
    old_width, old_height = old_size

    return int(pos_x*new_width/old_width), int(pos_y*new_height/old_height)


def scanlines_from_files(video, gaze, kwargs):
    SLITSAN_SOURCE = kwargs['source']
    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    VERTICAL_CROP = kwargs['VERTICAL_CROP']
    VERTICAL_FOCUS = kwargs['VERTICAL_FOCUS']
    off_center = int((1-VERTICAL_FOCUS)*LINE_HEIGHT)

    for num_frame in range(int(video.videoCaptureFrameCount)):
        img = video.getFrame(num_frame)
        pos_x, pos_y = gaze.getGaze(num_frame)

        if SLITSAN_SOURCE == 'center':
            img = transform_frame(img, LINE_HEIGHT)
            pos_x, pos_y = transform_gaze((pos_x, pos_y), new_size=(img.shape[1], img.shape[0]), old_size=(video.videoHeight, video.videoWidth))

            center_x = int(img.shape[1] // 2) 
            scanline = img[:,center_x-LINE_WIDTH:center_x+LINE_WIDTH+1]
            yield scanline

        elif SLITSAN_SOURCE == 'gaze-local':
            if pos_x >= LINE_WIDTH and pos_x < img.shape[1]-LINE_WIDTH and pos_y >= VERTICAL_CROP and pos_y < img.shape[0]-VERTICAL_CROP:
                center_y = int(img.shape[0] // 2) 
                scanline = img[center_y-VERTICAL_CROP: center_y+VERTICAL_CROP, pos_x-LINE_WIDTH:pos_x+LINE_WIDTH+1]
                scanline = cv.resize(scanline, (2*LINE_WIDTH+1, LINE_HEIGHT), interpolation=cv.INTER_CUBIC)
                yield scanline

        elif SLITSAN_SOURCE == 'gaze-global':
            img = transform_frame(img, LINE_HEIGHT)
            pos_x, pos_y = transform_gaze((pos_x, pos_y), new_size=(img.shape[1], img.shape[0]), old_size=(video.videoHeight, video.videoWidth))

            if pos_x>= LINE_WIDTH and pos_x < img.shape[1]-LINE_WIDTH:
                scanline = img[:, pos_x-LINE_WIDTH:pos_x+LINE_WIDTH+1]
                scanline[:pos_y-off_center] = scanline[:pos_y-off_center] * 0.7
                scanline[pos_y+off_center:] = scanline[pos_y+off_center: ] * 0.7
                yield scanline
        else:
            raise ValueError(f'Unknown slitscan source: {SLITSAN_SOURCE}')



def scanlines_from_Pupil_device(device, kwargs):
    LINE_WIDTH = kwargs['LINE_WIDTH']
    LINE_HEIGHT = kwargs['LINE_HEIGHT']
    VERTICAL_FOCUS = kwargs['VERTICAL_FOCUS']
    #SAMPLING = kwargs['SAMPLING']
    off_center = int((1-VERTICAL_FOCUS)*LINE_HEIGHT)

    while True:
        scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
        dt = scene_sample.timestamp_unix_seconds
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
        yield dt, scanline
