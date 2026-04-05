import sys
# This trick "hides" tensorflow from Mediapipe so it doesn't crash
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter 

# 1. LOAD THE TRAINED BRAIN
with open('sign_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# 2. SETUP MEDIAPIPE
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# Automatically load actions (A, B, C, D, E) saved from your training script
actions = model_dict['actions']

def extract_keypoints(results):
    # --- NEW WRIST-NORMALIZATION MATH ---
    # We must subtract the wrist (Landmark 0) from every other point
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    elif results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        # If no hand is detected, return zeros
        rh = np.zeros(21*3)
    return rh

# --- STABILITY SETTINGS ---
prediction_history = []
stability_threshold = 10 

# 3. LIVE FEED LOOP
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks for visual feedback
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- PREDICTION LOGIC ---
        keypoints = extract_keypoints(results).reshape(1, -1) 
        
        # Check if the hand is actually in frame (not just zeros)
        if np.any(keypoints):
            # 1. Get the raw prediction from the Neural Network
            prediction = model.predict(keypoints)[0]
            prediction_history.append(prediction)
            
            # 2. Keep the history at exactly 10 frames
            prediction_history = prediction_history[-stability_threshold:]
            
            # 3. The "Voting" Logic
            most_common = Counter(prediction_history).most_common(1)[0][0]
            predicted_char = actions[most_common]
            
            # Display the result on screen
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15,45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            prediction_history = []

        cv2.imshow('Sign Language Detector (Normalized)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()