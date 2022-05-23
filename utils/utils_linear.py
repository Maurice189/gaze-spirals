import cv2 as cv
import numpy as np
import time


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


def extract_scanline(img, gaze, kwargs):
    SLITSAN_SOURCE = kwargs['source']
    line_width = kwargs['line_width']
    line_height = kwargs['line_height']
    vertical_crop = kwargs['vertical_crop']
    vertical_focus = kwargs['vertical_focus']

    off_center = int((1-vertical_focus)*line_height)
    pos_x, pos_y = gaze
    old_size = img.shape[:2]

    if SLITSAN_SOURCE == 'center':
        img = transform_frame(img, line_height)
        pos_x, pos_y = transform_gaze((pos_x, pos_y), new_size=(img.shape[1], img.shape[0]), old_size=old_size)
        center_x = int(img.shape[1]//2) 
        scanline = img[:,center_x-line_width:center_x+line_width+1]
        return scanline

    elif SLITSAN_SOURCE == 'gaze-local':
        if pos_x >= line_width and pos_x < img.shape[1]-line_width and pos_y >= vertical_crop and pos_y < img.shape[0]-vertical_crop:
            center_y = int(img.shape[0] // 2) 
            scanline = img[center_y-vertical_crop: center_y+vertical_crop, pos_x-line_width:pos_x+line_width+1]
            scanline = cv.resize(scanline, (2*line_width+1, line_height), interpolation=cv.INTER_CUBIC)
            return scanline
        else:
            return np.zeros((line_height, 2*line_width+1, 3), np.uint8)

    elif SLITSAN_SOURCE == 'gaze-global':
        img = transform_frame(img, line_height)
        pos_x, pos_y = transform_gaze((pos_x, pos_y), new_size=(img.shape[1], img.shape[0]), old_size=old_size)
        if pos_x >= line_width and pos_x < img.shape[1]-line_width:
            scanline = img[:, pos_x-line_width:pos_x+line_width+1]
            scanline[:pos_y-off_center] = scanline[:pos_y-off_center] * 0.7
            scanline[pos_y+off_center:] = scanline[pos_y+off_center: ] * 0.7
            return scanline
        else:
            return np.zeros((line_height, 2*line_width+1, 3), np.uint8)
    else:
        raise ValueError(f'Unknown slitscan source: {SLITSAN_SOURCE}')


def scanlines_from_files(video, gaze, kwargs):
    dt = time.time()

    num_patches = 3
    canvas = np.zeros((video.videoHeight, video.videoWidth*num_patches, 3), dtype=np.uint8)

    for num_frame in range(int(video.videoCaptureFrameCount)):
        img = video.getFrame(num_frame)
        pos = gaze.getGaze(num_frame)
        dt += 1./video.videoCaptureFrameCount

        pos_x, pos_y = pos
        line_width = 8
        if num_frame %  num_patches == num_patches // 2:
            #img = cv.line(img, (pos_x, 0), (pos_x, video.videoHeight), color=(0, 0, 255), thickness=10, lineType=cv.LINE_AA)

            img[:, :pos_x-line_width] = (img[:, :pos_x-line_width]*.4).astype(np.uint8)
            img[:, pos_x+line_width:] = (img[:, pos_x+line_width:]*.4).astype(np.uint8)
            img[:, pos_x-line_width: pos_x+line_width+1] = (img[:, pos_x-line_width: pos_x+line_width+1] * 1.1).astype(np.uint8)
            
            img = cv.circle(img, (pos_x, 15), radius=15, color=(0, 255, 255), thickness=-1, lineType=cv.LINE_AA)
            img = cv.circle(img, (pos_x, video.videoHeight-15), radius=15, color=(0, 255, 255), thickness=-1, lineType=cv.LINE_AA)


        canvas[:, (num_frame%num_patches)*video.videoWidth: ((num_frame%num_patches)+1)*video.videoWidth] = img
        scaled = cv.resize(canvas, (int(400*canvas.shape[1]/canvas.shape[0]), 400))

        cv.imshow('gaze patch', scaled)
        cv.waitKey(1)

        yield int(dt), extract_scanline(img, pos, kwargs)


def scanlines_from_pupil_device(device, kwargs):
    while True:
        scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
        dt = scene_sample.timestamp_unix_seconds
        img = scene_sample.bgr_pixels
        yield dt, extract_scanline(img, (gaze_sample.x, gaze_sample.y), kwargs)