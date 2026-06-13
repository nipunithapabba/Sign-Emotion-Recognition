import sys
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter

# --- LOAD BOTH MODELS ---
with open('sign_model.p', 'rb') as f:
    sign_dict = pickle.load(f)
sign_model = sign_dict['model']
actions = sign_dict['actions']

with open('emotion_model.p', 'rb') as f:
    emotion_dict = pickle.load(f)
emotion_model = emotion_dict['model']
emotions = emotion_dict['emotions']

# --- MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_sign_keypoints(results):
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                         for lm in results.right_hand_landmarks.landmark]).flatten()
    elif results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                         for lm in results.left_hand_landmarks.landmark]).flatten()
    return np.zeros(21 * 3)

def extract_face_keypoints(results):
    if results.face_landmarks:
        nose = results.face_landmarks.landmark[1]
        return np.array([
            [lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]
            for lm in results.face_landmarks.landmark
        ]).flatten()
    return np.zeros(468 * 3)

def draw_confidence_bar(image, x, y, label, confidence, color):
    """
    Draws a labelled confidence bar.
    Full bar = 100% confidence, empty = 0%.
    """
    bar_max_width = 200
    bar_height = 18
    filled = int(confidence * bar_max_width)

    # Background
    cv2.rectangle(image, (x, y), (x + bar_max_width, y + bar_height), (50, 50, 50), -1)
    # Filled portion
    cv2.rectangle(image, (x, y), (x + filled, y + bar_height), color, -1)
    # Label + percentage
    cv2.putText(image, f'{label}: {confidence*100:.1f}%',
                (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- LIVE LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Diagnostic running... Press Q to quit.")
print("Watch the confidence bars — you want the top bar to be clearly dominant.\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Dark overlay panel on the right side for diagnostics
        cv2.rectangle(image, (0, 0), (640, 480), (0, 0, 0), -1)  # black bg
        # Rerender the camera feed on left half
        # (we'll draw everything as text on a black background for clarity)

        # ---- SIGN MODEL DIAGNOSTICS ----
        sign_kp = extract_sign_keypoints(results).reshape(1, -1)

        cv2.putText(image, '=== SIGN MODEL ===', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 117, 16), 2)

        if np.any(sign_kp):
            probs = sign_model.predict_proba(sign_kp)[0]
            top5_idx = np.argsort(probs)[::-1][:5]  # Top 5 predictions

            for i, idx in enumerate(top5_idx):
                label = actions[idx]
                conf = probs[idx]
                # Color: green if >70%, yellow if >40%, red if low
                color = (0, 200, 0) if conf > 0.7 else (0, 200, 200) if conf > 0.4 else (0, 0, 200)
                draw_confidence_bar(image, 20, 55 + i * 35, label, conf, color)

            best_sign = actions[top5_idx[0]]
            best_conf = probs[top5_idx[0]]
            cv2.putText(image, f'-> Best: {best_sign} ({best_conf*100:.1f}%)',
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No hand detected', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        # ---- EMOTION MODEL DIAGNOSTICS ----
        face_kp = extract_face_keypoints(results).reshape(1, -1)

        cv2.putText(image, '=== EMOTION MODEL ===', (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if np.any(face_kp):
            e_probs = emotion_model.predict_proba(face_kp)[0]
            for i, (emotion, prob) in enumerate(zip(emotions, e_probs)):
                color = (0, 200, 0) if prob > 0.7 else (0, 200, 200) if prob > 0.4 else (0, 0, 200)
                draw_confidence_bar(image, 20, 300 + i * 35, emotion, prob, color)

            best_emotion = emotions[np.argmax(e_probs)]
            best_econf = e_probs[np.argmax(e_probs)]
            cv2.putText(image, f'-> Best: {best_emotion} ({best_econf*100:.1f}%)',
                        (20, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No face detected', (20, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        cv2.imshow('Model Diagnostics', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()