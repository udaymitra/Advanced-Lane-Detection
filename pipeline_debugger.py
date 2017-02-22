from camera_calibration import CameraCalibrator
from image_utils import *
from lane_detection import LaneDetectionPipeline
from lane_finding_histogram import *
from lane_atrribute_computation import *
from lanes import *

cameraCalibrator = CameraCalibrator()
cameraCalibrator.restore('models/camera_calibration_model')
pipeline = LaneDetectionPipeline(cameraCalibrator)

img = read_image_as_rgb("test_images/test1.jpg")

binary_warped = pipeline.get_birds_eye_binary(img)
left_fit, right_fit, out_img = fit_left_and_right_lanes(binary_warped, 15, draw_rects=False)

lanes = Lanes()
lanes.add_current_lane_fits(left_fit, right_fit, binary_warped.shape[0], binary_warped.shape[1])

print("left_curvature: %f" % lanes.left_lane.radius_of_curvature)
print("right_curvature: %f" % lanes.right_lane.radius_of_curvature)
print("current_lane_width: %f" % lanes.current_lane_width)

img_with_lanes = lanes.draw_lanes(img, binary_warped)
# plt.imshow(img_with_lanes)
# plt.show()


# left_curvature, right_curvature, center_curvature = get_left_and_right_lane_curvature(left_fit, right_fit, img.shape[0], img.shape[1])
# print("left_curvature: %f, right_curvature: %f, center_curvature: %f"%(left_curvature, right_curvature, center_curvature))
#
# lane_width_mean, lane_width_std = get_lane_width_stats(left_fit, right_fit, img.shape[0], img.shape[1])
# print(lane_width_mean)
# print(lane_width_std)

# fig, ax = plt.subplots(1, 1)
# plot_left_and_right_lanes(binary_warped, 15, ax)
# plt.show()


from moviepy.editor import VideoFileClip

clip_output_file = 'marked_project_video.mp4'
clip = VideoFileClip("project_video.mp4")
clip_output = clip.fl_image(pipeline.process)
clip_output.write_videofile(clip_output_file, audio=False)


# input_file_dir = "temp_project_video_images/"
# output_file_dir = "out_project_video_images_fixed/"
# for idx in range(1000, 1260):
#     input_file_name = "image%d.jpg"%idx
#     input_file_path = input_file_dir + input_file_name
#     output_file_path = output_file_dir + input_file_name
#
#     img = read_image_as_rgb(input_file_path)
#     out_img = pipeline.process(img)
#     cv2.imwrite(output_file_path, out_img)
