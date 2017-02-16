import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import argparse

class CameraCalibrator:

    def __int__(self):
        self.mtx = None # distortion matrix
        self.dist = None # distortion coefficients

    def calibrate(self, chess_board_images_and_corners):
        print('calibrating camera...')
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        img_size = None

        for idx, (img, nx, ny) in enumerate(chess_board_images_and_corners):
            objp = np.zeros((ny * nx, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_size == None:
                h, w = gray.shape
                img_size = (w, h)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # draw corners and show image
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            # If found, add to corners and obj points arrays
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.mtx = mtx
        self.dist = dist

    def save(self, model_file):
        model = {"mtx": self.mtx, "dist": self.dist}
        pickle.dump(model, open(model_file, "wb"))
        print("calibration model saved at %s" % model_file)

    def restore(self, model_file):
        model = pickle.load(open(model_file, "rb"))
        self.mtx = model["mtx"]
        self.dist = model["dist"]
        print("calibration model restored from %s" % model_file)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

# def plot_single_camera_image_using_calibrator(img, nx, ny, calibrator):
    
# Helper method to visualize camera calibration
def plot_calibration_images(camera_images_and_corners, calibrator):
    print('plotting calibrated images')
    for idx, (img, nx, ny) in enumerate(camera_images_and_corners):
        fig, axes = plt.subplots(1, 4)
        fig.subplots_adjust(hspace=.5)

        ax1 = axes[0]
        ax1.axis("off")
        ax1.imshow(img)
        ax1.set_title("Original Image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        undist = calibrator.undistort(img)
        warped = None
        if ret == True:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            offset = 100  # offset for dst points
            img_size = (gray.shape[1], gray.shape[0])
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                              [img_size[0] - offset, img_size[1] - offset],
                              [offset, img_size[1] - offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)

            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)

        ax2 = axes[1]
        ax2.axis("off")
        ax2.imshow(img)
        ax2.set_title("Detected Corners")

        ax3 = axes[2]
        ax3.axis("off")
        ax3.imshow(undist)
        ax3.set_title("Corrected Image")

        if warped is not None:
            ax4 = axes[3]
            ax4.axis("off")
            ax4.imshow(warped)
            ax4.set_title("Warped Image")

        # plt.savefig("camera_cal_plots_warped/%d.png"%idx, dpi=200, pad_inches=0.5)
        plt.show()

def main():
    # resources
    CAMERA_IMG_NUM_AND_CORNERS = [
        (1, 9, 5),
        (2, 9, 6),
        (3, 9, 6),
        (4, 9, 6),
        (5, 9, 5),
        (6, 9, 6)]
    for img_num in range(7, 21):
        CAMERA_IMG_NUM_AND_CORNERS.append((img_num, 9, 6))

    CHESSBOARD_IMAGES_AND_CORNERS = [(cv2.imread('camera_cal/calibration%d.jpg' % file_num), nx, ny)
                                     for (file_num, nx, ny) in CAMERA_IMG_NUM_AND_CORNERS]

    # args
    parser = argparse.ArgumentParser(description='Calibrate camera using chessboard images')
    parser.add_argument("-i", "--inputmodelfile", help="path to input calibrated model file", type=str)
    parser.add_argument("-o", "--outputmodelfile", help="path to output calibrated model file", type=str)
    parser.add_argument("-p", "--plotcalibrations", help="if we should plot result", dest='plotcalibrations', action='store_true')
    parser.set_defaults(plotcalibrations=False)
    args = parser.parse_args()

    calibrator = CameraCalibrator()
    if args.inputmodelfile is None:
        calibrator.calibrate(CHESSBOARD_IMAGES_AND_CORNERS)
        calibrator.save(args.outputmodelfile)
    else:
        calibrator.restore(args.inputmodelfile)

    if args.plotcalibrations:
        plot_calibration_images(CHESSBOARD_IMAGES_AND_CORNERS, calibrator)

if __name__ == "__main__":
    main()