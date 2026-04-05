import sys
# This trick "hides" tensorflow from Mediapipe so it doesn't crash
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import os
import mediapipe as mp
import time

# 1. SETUP PATHS
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['F', 'G', 'H', 'I', 'J'])
no_sequences = 100 # 100 samples per letter
sequence_length = 1 # 1 frame per sample (static)

# Create the folders
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 2. MEDIAPIPE SETUP
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def extract_keypoints(results):
    if results.right_hand_landmarks:
        # Get the wrist (Landmark 0)
        wrist = results.right_hand_landmarks.landmark[0]
        # Subtract wrist from every finger so the data is "Relative"
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        # If no right hand, check the left hand
        if results.left_hand_landmarks:
            wrist = results.left_hand_landmarks.landmark[0]
            rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                          for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21*3)
            
    return rh # Only returning one hand's worth of data (63 numbers)

# 3. COLLECTION LOOP
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(no_sequences):
            ret, frame = cap.read()
            if not ret: break

            # Process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            # Draw landmarks so you can see if it's tracking well
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- SCREEN INSTRUCTIONS ---
            cv2.rectangle(image, (0,0), (640, 40), (0,0,0), -1)
            cv2.putText(image, f'COLLECTING: {action} | Sample: {sequence}', (15,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 2-second countdown before the very first sample of each letter
            if sequence == 0: 
                cv2.putText(image, 'GET READY...', (200,250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 4, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(2000)
            else: 
                cv2.imshow('Data Collection', image)

            # --- SAVE COORDINATES ---
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), '0.npy')
            np.save(npy_path, keypoints)

            # Short break between samples to adjust hand position
            # This makes the AI "smarter" about different angles
            cv2.waitKey(800) 

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

print("Data collection complete! You now have your dataset.")