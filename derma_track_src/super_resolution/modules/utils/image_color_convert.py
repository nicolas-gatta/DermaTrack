import cv2

def convert_image_RGB_to_YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def convert_image_YCrCb_to_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)