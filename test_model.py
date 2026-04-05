import sys
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter 
import time

# 1. LOAD THE BRAIN
with open('sign_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
actions = model_dict['actions']

# 2. SETUP MEDIAPIPE
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    elif results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    return rh

# --- TYPING & STABILITY SETTINGS ---
prediction_history = []
stability_threshold = 10 
sentence = []
current_char = ""
frames_held = 0
CHAR_LIMIT = 25 # How many frames to hold a sign before it "types" (Adjust this for speed!)

# --- TEXT CURSOR ---
last_blink_time = time.time()
show_cursor = True
cursor_speed = 0.5 # Blinks every 0.5 seconds

# 3. LIVE FEED LOOP
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- PREDICTION LOGIC ---
        keypoints = extract_keypoints(results).reshape(1, -1) 
        
        if np.any(keypoints):
            prediction = model.predict(keypoints)[0]
            prediction_history.append(prediction)
            prediction_history = prediction_history[-stability_threshold:]
            
            most_common = Counter(prediction_history).most_common(1)[0][0]
            predicted_char = actions[most_common]

            # --- TYPING LOGIC ---
            if predicted_char == current_char:
                frames_held += 1
            else:
                current_char = predicted_char
                frames_held = 0

            # If held long enough, add to sentence
            if frames_held == CHAR_LIMIT:
                sentence.append(predicted_char)
                frames_held = 0 # Reset to prevent double typing

            # Display "Holding" Progress
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15,45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Progress bar under the letter
            bar_width = int(frames_held * (250/CHAR_LIMIT))
            cv2.rectangle(image, (0, 60), (bar_width, 70), (0, 255, 0), -1)
        else:
            prediction_history = []
            frames_held = 0

        # --- SENTENCE DISPLAY WITH BLINKING CURSOR ---
        # 1. Background Bar
        cv2.rectangle(image, (0, 420), (640, 480), (245, 117, 16), -1)
        
        # 2. Blink Logic: Toggle show_cursor every 0.5 seconds
        if time.time() - last_blink_time > cursor_speed:
            show_cursor = not show_cursor
            last_blink_time = time.time()

        # 3. Build the display string
        display_text = "".join(sentence)
        if show_cursor:
            display_text += "|" # Our "Blinking" Cursor character
        
        # 4. Render to Screen
        cv2.putText(image, display_text, (20, 465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Sentence Builder', image)

        # Keys
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): sentence = [] # Clear everything
        elif key == ord(' '): sentence.append(" ") # Manual space
        elif key == 8: # Backspace key
            if len(sentence) > 0: sentence.pop()

cap.release()
cv2.destroyAllWindows()