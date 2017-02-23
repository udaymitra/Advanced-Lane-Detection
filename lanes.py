import numpy as np
import cv2
from perspective_transform import BirdsEyeViewTransform
from lane_finding_histogram import *

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
        self.best_fit_meters = None
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
        self.best_fit = np.polyfit(self.ally, self.bestx, 2)

        self.best_fit_meters = np.polyfit(self.ally * Y_M_PER_PIX, self.bestx * X_M_PER_PIX, 2)
        self.radius_of_curvature = ((1 + (2 * self.best_fit_meters[0] * IMG_HEIGHT * Y_M_PER_PIX + self.best_fit_meters[1]) ** 2) ** 1.5) / np.absolute(2 * self.best_fit_meters[0])

    def get_x(self, y):
        return self.current_fit[0] * y ** 2 + self.current_fit[1] * y + self.current_fit[2]

    def reset(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.best_fit_meters = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None

        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')

        self.allx = None
        self.ally = np.linspace(0, IMG_HEIGHT - 1, 20)

# get Lane width
def get_lane_width_stats(left_fit, right_fit, img_height, x_m_per_pix):
    y = np.linspace(0, img_height - 1, 15)
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    lane_widths = (right_x - left_x) * x_m_per_pix
    lane_width_mean = lane_widths.mean()
    lane_width_std = lane_widths.std()
    return (lane_width_mean, lane_width_std)

class Lanes():
    def __init__(self):
        self.birds_eye_transform = BirdsEyeViewTransform()

        self.left_lane = Line()
        self.right_lane = Line()

        self.recent_lane_widths = []
        self.current_lane_width = None
        self.car_meters_off_center = None
        self.lane_locked = False
        self.radius_of_curvature = None

    def is_new_fit_good(self, left_fit, right_fit, img_height):
        if left_fit is None or right_fit is None:
            return False

        lane_width_mean, lane_width_std = get_lane_width_stats(left_fit, right_fit, img_height, X_M_PER_PIX)
        if lane_width_std > 0.1 * lane_width_mean:
            # print("Error: lane width stddev %f is more than 10%% of lane width %f" % (lane_width_std, lane_width_mean))
            return False

        if self.recent_lane_widths:
            recent_lane_width_mean = np.mean(self.recent_lane_widths)
            if abs(lane_width_mean - recent_lane_width_mean) > 0.1 * recent_lane_width_mean:
                # print("Error: current lane width %f is far from from mean %f" % (lane_width_mean, recent_lane_width_mean))
                return False

        return True

    def add_current_lane_fits(self, binary_warped):
        img_height = binary_warped.shape[0]
        img_width = binary_warped.shape[1]

        self.lane_locked = False
        left_fit = None
        right_fit = None

        if self.left_lane.detected and self.right_lane.detected:
            left_fit, right_fit = fit_left_and_right_lanes_using_last_frame_info(binary_warped, self.left_lane.best_fit, self.right_lane.best_fit)
            self.lane_locked = self.is_new_fit_good(left_fit, right_fit, img_height)

        if not self.lane_locked:
            left_fit, right_fit, _ = fit_left_and_right_lanes(binary_warped, 15, draw_rects=False)

        self.left_lane.add_new_lane_info(left_fit)
        self.right_lane.add_new_lane_info(right_fit)

        lane_x_diff = (self.right_lane.bestx - self.left_lane.bestx) * X_M_PER_PIX
        self.current_lane_width = lane_x_diff.mean()

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

        self.radius_of_curvature = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature) / 2

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

        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(self.recent_lane_widths) > 1:
            if self.lane_locked:
                cv2.putText(result, "Lane Locked", (550, 40), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(result, "Lane Lost", (550, 40), font, 1, (255, 0, 0), 2)

        cv2.putText(result, "Radius of Curvature: %d m" % int(self.radius_of_curvature) , (400, 70),
                    font, 1, (255, 255, 0), 2)
        cv2.putText(result, "Distance from center: %.2f m" % np.abs(self.car_meters_off_center), (400, 100),
                    font, 1, (255, 255, 255), 2)
        return result

    def reset(self):
        self.left_lane = Line()
        self.right_lane = Line()

        self.recent_lane_widths = []
        self.current_lane_width = None
        self.car_meters_off_center = None
        self.lane_locked = False
        self.radius_of_curvature = None