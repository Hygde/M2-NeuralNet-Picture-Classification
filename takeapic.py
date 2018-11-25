import cv2
import numpy as np
from camera import Camera

class TakeAPic(Camera):

    def __init__(self):
        super(TakeAPic,self).__init__()

    def getPicture(self):
        continuer = True
        pic = None
        while continuer:
            pic = self.getFrameFromCamera()
            pic = cv2.resize(pic, (350,350))
            cv2.imshow("picviewer",pic)
            if cv2.waitKey(33) == ord(' '):continuer = False
        return pic

    def picFormatter(self, pic, shape, channels):
        return np.reshape(pic, [1,shape[0],shape[1],channels])