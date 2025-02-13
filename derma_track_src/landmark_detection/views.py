from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .services.pose_estimator import detect_body_part

import cv2
# Create your views here.

def video_stream(request):

    if request.headers.get('HX-Request'):
        return render(request, 'partial/video_stream.html')
    

    
def landmark_detection_view(request):
    """Opens the camera, detects landmarks, and sends JSON response when 'q' is pressed."""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JsonResponse({"error": "Could not open camera"}, status=500)

    detected_body_part = "No detection"
    all_landmarks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the current frame
        detected_body_part, all_landmarks = detect_body_part(frame)

        # Display the detected body part and landmarks on the frame
        cv2.putText(frame, f"Detected: {detected_body_part}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmarks on the frame
        for landmark in all_landmarks.values():
            cv2.circle(frame, (landmark["x"], landmark["y"]), 5, (0, 0, 255), -1)

        # Show the live feed
        cv2.imshow("Live Detection (Press 'q' to exit)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Return the last detected body part and all landmarks as JSON
    return JsonResponse({"detected_part": detected_body_part})