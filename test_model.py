import sys
# Hide tensorflow to prevent Mediapipe conflict
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
    """Normalized 3D landmarks relative to the wrist."""
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

# --- SYSTEM SETTINGS ---
prediction_history = []
stability_threshold = 10 
sentence = []
current_char = ""
frames_held = 0
CHAR_LIMIT = 25 # Speed of typing (lower = faster)

# --- UI & SCROLLING SETTINGS ---
last_blink_time = time.time()
show_cursor = True
cursor_speed = 0.5 
full_transcript = []  # Every line completed
scroll_index = 0      # Current view position
MAX_LINES_VISIBLE = 3 # How many history lines to show

# 3. LIVE FEED LOOP
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
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

            if predicted_char == current_char:
                frames_held += 1
            else:
                current_char = predicted_char
                frames_held = 0

            if frames_held == CHAR_LIMIT:
                sentence.append(predicted_char)
                frames_held = 0 

            # Top UI (Live Recognition)
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15,45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            bar_width = int(frames_held * (250/CHAR_LIMIT))
            cv2.rectangle(image, (0, 60), (bar_width, 70), (0, 255, 0), -1)
        else:
            prediction_history = []
            frames_held = 0

        # --- SCROLLING DISPLAY LOGIC ---
        # Draw background bar (Height adjusted for history)
        cv2.rectangle(image, (0, 350), (640, 480), (245, 117, 16), -1)
        
        # Cursor Blink
        if time.time() - last_blink_time > cursor_speed:
            show_cursor = not show_cursor
            last_blink_time = time.time()

        display_text = "".join(sentence)
        
        # Auto-Scroll/New Line Trigger
        if len(display_text) > 25:
            full_transcript.append(display_text)
            sentence = [] 
            display_text = ""
            # Push view to bottom on new line
            scroll_index = max(0, len(full_transcript) - MAX_LINES_VISIBLE)

        # Get the visible slice of history
        visible_history = full_transcript[scroll_index : scroll_index + MAX_LINES_VISIBLE]

        # Render History (Lines above)
        for i, line in enumerate(visible_history):
            cv2.putText(image, line, (20, 380 + (i*30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA)

        # Render Active Line
        cursor = "|" if show_cursor else ""
        cv2.putText(image, display_text + cursor, (20, 465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Scroll Indicators
        if scroll_index > 0:
            cv2.putText(image, "^", (620, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if len(full_transcript) > scroll_index + MAX_LINES_VISIBLE:
            cv2.putText(image, "v", (620, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Sign Language Sentence Builder', image)

        # --- CONTROLS ---
        key = cv2.waitKey(10)
        if key == ord('q'): 
            break
        elif key == ord('c'): 
            sentence, full_transcript, scroll_index = [], [], 0
        elif key == ord(' '): 
            sentence.append(" ") 
        elif key == ord('.'): 
            sentence.append(".")
        elif key == 8: # Backspace
            if len(sentence) > 0: sentence.pop()
        
        # Arrow Keys (Windows) or 'w/s' for Scrolling
        elif key == 2490368 or key == ord('w'): 
            scroll_index = max(0, scroll_index - 1)
        elif key == 2621440 or key == ord('s'): 
            scroll_index = min(max(0, len(full_transcript) - MAX_LINES_VISIBLE), scroll_index + 1)

cap.release()
cv2.destroyAllWindows()