import cv2
from enum import Enum

class ImageColorConverter(int, Enum):
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    YCrCb2BGR = cv2.COLOR_YCrCb2BGR
    BGR2RGB = cv2.COLOR_BGR2RGB
    RGB2BGR = cv2.COLOR_RGB2BGR
        
class ImageConverter:
    @staticmethod
    def convert_image(image, mode: ImageColorConverter):
        return cv2.cvtColor(image, mode)