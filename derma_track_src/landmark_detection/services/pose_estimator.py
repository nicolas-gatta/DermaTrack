import cv2
import mediapipe as mp
import numpy as np
import base64
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

model_path = os.path.join(settings.BASE_DIR, "landmark_detection", "models", "pose_landmarker_heavy.task")

def decode_base64_image(base64_string):
    """
    Decode a Base64 string into an OpenCV image (numpy array).
    """
    try:
        img_data = base64.b64decode(base64_string, validate = True)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None
    
            
def detect_body_part(base64_image):
    """
    Detects the closest body part to the center of the frame.
    """
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    image = decode_base64_image(base64_image)
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_image)
        
        landmarks = pose_landmarker_result.pose_landmarks
        
        if landmarks:
            distances = {}

            min_dist = float('inf')
            nearest_part = None
            all_landmarks = {}

            for i, landmark in enumerate(landmarks[0]):
                x, y = int(landmark.x * width), int(landmark.y * height)
                all_landmarks[i] = {"x": x, "y": y}
                dist = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)

                if dist < min_dist:
                    min_dist = dist
                    nearest_part = i

            return body_parts.get(nearest_part, "Unknown")

            if landmarks[0].visibility > 0.5:  # Nose (Head)
                distances["Head"] = abs(landmarks[0].x * width - center_x)
            if landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:  # Shoulders
                distances["Shoulders"] = abs((landmarks[11].x + landmarks[12].x) / 2 * width - center_x)
            if landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5:  # Torso
                distances["Torso"] = abs((landmarks[23].x + landmarks[24].x) / 2 * width - center_x)
            if landmarks[15].visibility > 0.5 and landmarks[16].visibility > 0.5:  # Hands
                distances["Hands"] = abs((landmarks[15].x + landmarks[16].x) / 2 * width - center_x)
            if landmarks[27].visibility > 0.5 and landmarks[28].visibility > 0.5:  # Feet
                distances["Feet"] = abs((landmarks[27].x + landmarks[28].x) / 2 * width - center_x)

            if distances:
                return min(distances, key=distances.get)  # Return most centered body part
        return "Unknown"
