import matplotlib.pyplot as plt
import glob
from camera_calibration import CameraCalibrator
from perspective_transform import BirdsEyeViewTransform
from image_utils import *

class LaneDetectionPipeline:
    BIRDS_EYE_TRANSFORM = BirdsEyeViewTransform()

    def __init__(self, camera_calibrator):
        self.camera_calibrator = camera_calibrator

    def process(self, img):
        undist = self.camera_calibrator.undistort(img)
        gray = custom_gray_transform(undist)
        crop = trapezoidal_crop(gray)
        birds_eye_view = self.BIRDS_EYE_TRANSFORM.get_birds_eye_view(crop)

        return birds_eye_view

# debug code
def main():
    test_img_files = glob.glob("test_images/*.jpg")
    test_images = [read_image_as_rgb(f) for f in test_img_files]
    cameraCalibrator = CameraCalibrator()
    cameraCalibrator.restore('models/camera_calibration_model')
    pipeline = LaneDetectionPipeline(cameraCalibrator)

    img = test_images[0]
    undistorted = pipeline.process(img)
    bet = BirdsEyeViewTransform()
    birds_eye_undistorted = bet.get_birds_eye_view(undistorted)
    crop = trapezoidal_crop(undistorted)
    birds_eye_crop = bet.get_birds_eye_view(crop)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].imshow(undistorted, cmap='gray')
    # ax[0][0].set_title("hls and gradient transform")
    # ax[0][0].axis("off")
    # ax[0][1].imshow(birds_eye_undistorted, cmap='gray')
    # ax[0][1].set_title("birds eye")
    # ax[0][1].axis("off")

    # ax[1][0].imshow(crop, cmap='gray')
    # ax[1][0].title("trapeziodal crop")
    # ax[1][0].axis("off")
    # ax[1][1].imshow(birds_eye_crop, cmap='gray')
    # ax[1][1].title("birds eye crop")
    # ax[1][1].axis("off")

    histogram = np.sum(birds_eye_crop, axis=0)
    plt.plot(histogram)
    plt.show()

if __name__ == "__main__":
    main()
