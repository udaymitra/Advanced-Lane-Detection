from camera_calibration import CameraCalibrator
from image_utils import *
from lane_detection import LaneDetectionPipeline
from lane_finding_histogram import *
from lanes import *
from glob import glob

cameraCalibrator = CameraCalibrator()
cameraCalibrator.restore('models/camera_calibration_model')
pipeline = LaneDetectionPipeline(cameraCalibrator)

test_images_dir = "test_images"
output_images_dir = "output_images"
input_file_paths = glob(test_images_dir + "/*.jpg")
for input_file_path in input_file_paths:
    output_file_path = input_file_path.replace(test_images_dir, output_images_dir)
    img = read_image_as_rgb(input_file_path)
    out_img = pipeline.process(img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_path, out_img)
    pipeline.reset()

from moviepy.editor import VideoFileClip
test_videos_dir = "test_videos"
output_videos_dir = "output_videos"
video_file_paths = glob(test_videos_dir + "/*.mp4")
for video_file_path in video_file_paths:
    output_file_path = video_file_path.replace(test_videos_dir, output_videos_dir)
    clip = VideoFileClip(video_file_path)
    clip_output = clip.fl_image(pipeline.process)
    clip_output.write_videofile(output_file_path, audio=False)
    pipeline.reset()