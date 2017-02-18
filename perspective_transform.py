import numpy as np
import cv2

class BirdsEyeViewTransform:
    SRC_COORDS=np.float32([(190,720),(1130,720),(725,460),(570,460)])
    DEST_COORDS=np.float32([(265,720),(1130,720),(1130,0),(265,0)])
    M = cv2.getPerspectiveTransform(SRC_COORDS, DEST_COORDS)
    Minv = cv2.getPerspectiveTransform(DEST_COORDS, SRC_COORDS)

    def get_birds_eye_view(self, gray):
        return cv2.warpPerspective(gray, self.M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

    def get_original_view(self, birds_eye_view):
        return cv2.warpPerspective(birds_eye_view, self.Minv, (birds_eye_view.shape[1], birds_eye_view.shape[0]),
                                   flags=cv2.INTER_LINEAR)
