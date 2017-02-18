import cv2

def read_image_as_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)