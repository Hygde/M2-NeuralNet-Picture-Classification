import cv2
import logging

class Camera():

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.debug("Creating camera class")
        self.cam = cv2.VideoCapture(0)
        
    def getFrameFromCamera(self):
        self.logger.debug("Getting data from the camera")
        ret, frame = self.cam.read()
        #self.logger.debug(frame)
        if(not ret):
            self.logger.debug("An error occurs while reading the video")
            frame = None
        else:self.logger.debug("The frame is correctly read")
        return frame
        
    def closeCamera(self):
        self.logger.debug("Closing the camera")
        self.cam.release()
