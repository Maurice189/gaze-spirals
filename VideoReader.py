import cv2 as cv
import numpy as np

class VideoReader():
    """This class reads video data"""
    def __init__(self, filename):
        self.videoCapture = cv.VideoCapture(filename)  
        # get frames per second
        self.videoCaptureFps = self.videoCapture.get(cv.CAP_PROP_FPS)
        # get total frame count
        self.videoCaptureFrameCount = int(self.videoCapture.get(cv.CAP_PROP_FRAME_COUNT))
        self.videoHeight = int(self.videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.videoWidth = int(self.videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.lastFrame=0
        self.filename = filename

    def getFrame(self,fnr):
        """
        This function retrieves a video image by frame number
        """
        if self.lastFrame != fnr-1:
            self.videoCapture.set(cv.CAP_PROP_POS_FRAMES, fnr)
        success,frame = self.videoCapture.read()
        self.lastFrame=fnr

        if not success:
            frame = np.zeros((self.videoHeight, self.videoWidth, 3), np.uint8)
            self.lastFrame=0
        return frame
