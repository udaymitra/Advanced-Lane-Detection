import numpy as np
import cv2

def read_image_as_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def normalize(sobel):
    sobel = np.absolute(sobel)
    return (sobel * 255. / sobel.max()).astype(np.uint8)

# Use HLS color space and separate out L, S channels
# On L-channel use sobel on X-space to identify vertical lines
# Use S-channel to identify where there is max saturation
# This helps identify yellow lanes on road better
def hls_transform(img, s_thresh, sx_thresh, kernel_size):
    img = np.copy(img)

    # Convert to HLS color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)  # Take the derivative in x
    scaled_sobel_x = normalize(sobelx) # Absolute x derivative to accentuate lines away from horizontal

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return (sxbinary, s_binary)

# Convert to single channel binary image
# Uses HLS color transformation and gradient sobel
def custom_gray_transform(img, s_thresh=(170, 255), sx_thresh=(20, 100), kernel_size =3):
    sxbinary, s_binary = hls_transform(img, s_thresh, sx_thresh, kernel_size)
    out_img = np.zeros_like(sxbinary)
    out_img[(sxbinary == 1) | (s_binary == 1)] = 1
    return out_img

def get_hls_transform_image(img, s_thresh=(170, 255), sx_thresh=(20, 100), kernel_size =3):
    sxbinary, s_binary = hls_transform(img, s_thresh, sx_thresh, kernel_size)

    # Stack each channel
    # first channel is an all black image
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

# Build a mask thats approximately a trapeziod that can contain the lane infront of the car
# Apply mask and only retain that part of the image
def trapezoidal_crop(gray_img):
    H, W = gray_img.shape[:2]
    trapezoidal_aoi = np.array([[
        (40, H),
        (W / 2 - 40, H / 2 + 60),
        (W / 2 + 40, H / 2 + 60),
        (W - 40, H)]], dtype=np.int32)

    mask = np.zeros_like(gray_img, dtype=np.int32)
    cv2.fillPoly(mask, trapezoidal_aoi, 1)
    return np.uint8(gray_img & mask)