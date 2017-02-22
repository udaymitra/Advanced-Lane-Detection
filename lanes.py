import numpy as np
import cv2
from perspective_transform import BirdsEyeViewTransform

# Define conversions in x and y from pixels space to meters
Y_M_PER_PIX = 30/720 # meters per pixel in y dimension
X_M_PER_PIX = 3.7/700 # meters per pixel in x dimension

# Number of entries to store
LAST_N = 5
IMG_HEIGHT = 720

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = np.linspace(0, IMG_HEIGHT - 1, 20)

    def add_new_lane_info(self, current_fit):
        self.detected = True
        self.current_fit = current_fit

        # get all Xs
        self.allx = current_fit[0] * self.ally ** 2 + current_fit[1] * self.ally + current_fit[2]

        # retain the last N x fitted values
        if (len(self.recent_xfitted) < LAST_N):
            self.recent_xfitted.append(self.allx)
        else:
            self.recent_xfitted[:-1] = self.recent_xfitted[1:]
            self.recent_xfitted[-1] = self.allx

        # compute best X
        self.bestx = np.mean(self.recent_xfitted, axis=0)

        self.best_fit = np.polyfit(self.ally * Y_M_PER_PIX, self.bestx * X_M_PER_PIX, 2)
        self.radius_of_curvature = ((1 + (2 * self.best_fit[0] * IMG_HEIGHT * Y_M_PER_PIX + self.best_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.best_fit[0])

    def get_x(self, y):
        return self.current_fit[0] * y ** 2 + self.current_fit[1] * y + self.current_fit[2]

    def reset(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None

        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')

        self.allx = None
        self.ally = np.linspace(0, IMG_HEIGHT - 1, 20)

class Lanes():
    def __init__(self):
        self.birds_eye_transform = BirdsEyeViewTransform()

        self.left_lane = Line()
        self.right_lane = Line()

        self.recent_lane_widths = []
        self.current_lane_width = None
        self.car_meters_off_center = None
        self.text = ''

    def add_current_lane_fits(self, left_fit, right_fit, img_height, img_width):
        self.text = ''
        self.left_lane.add_new_lane_info(left_fit)
        self.right_lane.add_new_lane_info(right_fit)

        lane_x_diff = (self.right_lane.bestx - self.left_lane.bestx) * X_M_PER_PIX
        self.current_lane_width = lane_x_diff.mean()
        lane_width_std = lane_x_diff.std()
        if lane_width_std > 0.1 * self.current_lane_width:
            self.text += "Error: lane width stddev %f is more than 10%% of lane width %f" % (lane_width_std, self.current_lane_width)
            print("ERROR 1")

        if self.recent_lane_widths:
            recent_lane_width_mean = np.mean(self.recent_lane_widths)
            if abs(self.current_lane_width - recent_lane_width_mean) > 0.1 * recent_lane_width_mean:
                self.text += "Error: current lane width %f is far from from mean %f" % (self.current_lane_width, recent_lane_width_mean)
                print("ERROR 2")
                print(self.text)

        # retain the last N lane widths
        if (len(self.recent_lane_widths) < LAST_N):
            self.recent_lane_widths.append(self.current_lane_width)
        else:
            self.recent_lane_widths[:-1] = self.recent_lane_widths[1:]
            self.recent_lane_widths[-1] = self.current_lane_width

        left_lane_start = self.left_lane.get_x(img_height - 1)
        right_lane_start = self.right_lane.get_x(img_height - 1)
        screen_middel_pixel = img_width / 2
        car_middle_pixel = int((right_lane_start + left_lane_start) / 2)
        screen_off_center = screen_middel_pixel - car_middle_pixel
        self.car_meters_off_center = X_M_PER_PIX * screen_off_center

        radius_of_curvature = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature) / 2
        self.text += "curvature: %d m, %s of center: %.2f" % (int(radius_of_curvature),
                                                        "left" if self.car_meters_off_center > 0 else "right",
                                                        np.abs(self.car_meters_off_center))

    def draw_lanes(self, undist, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_lane.bestx, self.left_lane.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.bestx, self.right_lane.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane polygon onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        center_lane_bestx = (self.left_lane.bestx + self.right_lane.bestx) / 2
        pts_center = np.array([np.flipud(np.transpose(np.vstack([center_lane_bestx, self.right_lane.ally])))])
        # Draw left, right and center lanes
        for pts in [pts_left, pts_right, pts_center]:
            cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(255, 0, 0), thickness=20)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.birds_eye_transform.get_original_view(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        result = cv2.putText(result, self.text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (255, 255, 0), 2)
        return result

    def reset(self):
        self.left_lane = Line()
        self.right_lane = Line()

        self.recent_lane_widths = []
        self.current_lane_width = None
        self.car_meters_off_center = None
        self.text = ''