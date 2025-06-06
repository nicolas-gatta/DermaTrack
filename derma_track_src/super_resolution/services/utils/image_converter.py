import cv2
from enum import Enum

class ImageColorConverter(int, Enum):
    """
    An enumeration of color conversion codes for image processing using OpenCV.
    Attributes:
        BGR2YCrCb: Converts an image from BGR color space to YCrCb color space.
        YCrCb2BGR: Converts an image from YCrCb color space to BGR color space.
        BGR2RGB: Converts an image from BGR color space to RGB color space.
        RGB2BGR: Converts an image from RGB color space to BGR color space.
    """
    
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    YCrCb2BGR = cv2.COLOR_YCrCb2BGR
    BGR2RGB = cv2.COLOR_BGR2RGB
    RGB2BGR = cv2.COLOR_RGB2BGR
        
class ImageConverter:
    @staticmethod
    def convert_image(image, mode: ImageColorConverter):
        return cv2.cvtColor(image, mode)