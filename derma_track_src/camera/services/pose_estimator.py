import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from django.conf import settings

body_parts = {
    0: "Nose", 1: "Left Eye (Inner)", 2: "Left Eye", 3: "Left Eye (Outer)",
    4: "Right Eye (Inner)", 5: "Right Eye", 6: "Right Eye (Outer)",
    7: "Left Ear", 8: "Right Ear", 9: "Mouth (Left)", 10: "Mouth (Right)",
    11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow",
    14: "Right Elbow", 15: "Left Wrist", 16: "Right Wrist",
    17: "Left Pinky", 18: "Right Pinky", 19: "Left Index",
    20: "Right Index", 21: "Left Thumb", 22: "Right Thumb",
    23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle", 29: "Left Heel", 30: "Right Heel",
    31: "Left Foot Index", 32: "Right Foot Index"
}

model_path = os.path.join(settings.BASE_DIR, "camera", "models", "pose_landmarker_heavy.task")

def decode_base64_image(base64_string: str):
    """
    Decode a Base64 string into an OpenCV image.

    Args:
        base64_string (str): Base64-encoded image string.

    Returns:
        np.ndarray or None: Decoded OpenCV image or None.
    """
    
    try:
        img_data = base64.b64decode(base64_string, validate = True)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None
    
            
def detect_body_part(base64_image: str):
    """
    Detects the body part closest to the center of the image using a pose model.

    Args:
        base64_image (str): Base64-encoded image string.

    Returns:
        str: Name of the closest body part to the image center, or "Unknown" if detection fails.
    """
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    image = decode_base64_image(base64_image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_image)
        
        landmarks = pose_landmarker_result.pose_landmarks
        
        if landmarks:
            
            distances = {}

            if distances:
                return min(distances, key=distances.get)  # Return most centered body part
        return "Unknown"
