import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

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
            
def detect_body_part(frame):
    """Detects the closest body part to the center of the frame."""
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return "No body detected", {}

        landmarks = results.pose_landmarks.landmark
        min_dist = float('inf')
        nearest_part = None
        all_landmarks = {}

        for i, landmark in enumerate(landmarks):
            x, y = int(landmark.x * width), int(landmark.y * height)
            all_landmarks[i] = {"x": x, "y": y}
            dist = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_part = i

        return body_parts.get(nearest_part, "Unknown Body Part"), all_landmarks
