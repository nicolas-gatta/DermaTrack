import cv2

def convert_image_BGR_to_YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def convert_image_YCrCb_to_BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)