import cv2

class ImageConverter:
    
    _image_convert_dict = {
        "BGR_to_YCrCb": cv2.COLOR_BGR2YCrCb,
        "YCrCb_to_BGR": cv2.COLOR_BGR2YCrCb,
        "BGR_to_RGB": cv2.COLOR_BGR2RGB
    }
            
    @staticmethod
    def convert_image(image, mode):

        if mode in ImageConverter.image_convert_dict:
            return cv2.cvtColor(image, ImageConverter.image_convert_dict[mode])
        else:
            raise ValueError(f"Failed to find the value: {mode}")