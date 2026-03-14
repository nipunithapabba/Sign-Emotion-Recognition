import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic (The 'all-in-one' model)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Start the Webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Prepare image (MediaPipe needs RGB, OpenCV uses BGR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Draw Landmarks for Emotion (Face)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        # 3. Draw Landmarks for Sign (Hands)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Show the result
        cv2.imshow('Sign & Emotion Tracker', image)

        # Break loop with 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()