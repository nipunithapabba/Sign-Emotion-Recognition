import sys
# This trick "hides" tensorflow from Mediapipe so it doesn't crash
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import mediapipe as mp
import pickle

# 1. LOAD THE TRAINED BRAIN
with open('sign_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# 2. SETUP MEDIAPIPE
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F']) 

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# 3. LIVE FEED LOOP
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks for visual feedback
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- PREDICTION LOGIC ---
        keypoints = extract_keypoints(results).reshape(1, -1) # Format for Scikit-Learn
        
        # Only predict if at least one hand is visible
        if np.any(keypoints):
            prediction = model.predict(keypoints)
            predicted_char = actions[prediction[0]]
            
            # Display the result on screen
            cv2.rectangle(image, (0,0), (200, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15,45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()