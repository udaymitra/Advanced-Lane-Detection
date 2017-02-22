import numpy as np
import matplotlib.pyplot as plt

def temp():
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                  for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images

    plt.show()

    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')


# Define conversions in x and y from pixels space to meters
Y_M_PER_PIX = 30/720 # meters per pixel in y dimension
X_M_PER_PIX = 3.7/700 # meters per pixel in x dimension

def get_left_and_right_lane_curvature(left_fit, right_fit, img_height, img_width):
    y, left_fit_cr, right_fit_cr, center_fit_cr = get_lanes_fit_in_meters(left_fit, right_fit, img_height)

    left_curverad = ((1 + (2*left_fit_cr[0]*img_height*Y_M_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*img_height*Y_M_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    center_curverad = ((1 + (2 * center_fit_cr[0] * img_height * Y_M_PER_PIX + center_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * center_fit_cr[0])

    return (left_curverad, right_curverad, center_curverad)

def get_lanes_fit_in_meters(left_fit, right_fit, img_height):
    y = np.linspace(0, img_height - 1, 15)
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    left_x = left_x[::-1]  # Reverse to match top-to-bottom in y
    right_x = right_x[::-1]  # Reverse to match top-to-bottom in y

    left_fit_cr = np.polyfit(y * Y_M_PER_PIX, left_x * X_M_PER_PIX, 2)
    right_fit_cr = np.polyfit(y * Y_M_PER_PIX, right_x * X_M_PER_PIX, 2)
    center_fit_cr = np.polyfit(y * Y_M_PER_PIX, ((left_x + right_x) / 2) * X_M_PER_PIX, 2)

    return (y, left_fit_cr, right_fit_cr, center_fit_cr)

def get_lane_width_stats(left_fit, right_fit, img_height, img_width):
    y = np.linspace(0, img_height - 1, 15)
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    lane_widths = (right_x - left_x) * X_M_PER_PIX
    lane_width_mean = lane_widths.mean()
    lane_width_std = lane_widths.std()
    return (lane_width_mean, lane_width_std)