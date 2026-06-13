import sys
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter
import time
import pyttsx3
import queue

# --- TTS SETUP ---
speech_queue = queue.Queue()

def speak(text):
    """Add text to the speech queue (strips dots/spaces so it sounds natural)."""
    clean = text.strip(" .")
    if clean:
        speech_queue.put(clean)

def process_speech():
    """Reinitializes engine fresh each time — Windows-safe fix."""
    if not speech_queue.empty():
        text_to_say = speech_queue.get()
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text_to_say)
        engine.runAndWait()
        engine.stop()

# --- LOAD SIGN MODEL ---
with open('sign_model.p', 'rb') as f:
    sign_dict = pickle.load(f)
sign_model = sign_dict['model']
actions = sign_dict['actions']

# --- LOAD EMOTION MODEL ---
with open('emotion_model.p', 'rb') as f:
    emotion_dict = pickle.load(f)
emotion_model = emotion_dict['model']
emotions = emotion_dict['emotions']

# Emotion display colours (BGR) — each emotion gets its own colour
EMOTION_COLORS = {
    'Happy':     (0, 215, 255),   # Gold
    'Sad':       (255, 100, 50),  # Blue
    'Angry':     (0, 0, 220),     # Red
    'Neutral':   (180, 180, 180), # Grey
    'Surprised': (0, 255, 180),   # Green-teal
}

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_sign_keypoints(results):
    """Normalized 3D hand landmarks relative to the wrist."""
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    elif results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        rh = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
    return rh

def extract_face_keypoints(results):
    """
    Normalized face landmarks with translation + rotation correction.
    Fixes the head-tilt problem — tilting up/down no longer affects emotion prediction.
    """
    if not results.face_landmarks:
        return np.zeros(468 * 3)

    landmarks = results.face_landmarks.landmark

    # Step 1: Translate relative to nose tip
    nose = landmarks[1]
    coords = np.array([[lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]
                       for lm in landmarks])

    # Step 2: Rotate to correct for head tilt using eye line
    left_eye  = np.array([landmarks[33].x  - nose.x, landmarks[33].y  - nose.y])
    right_eye = np.array([landmarks[263].x - nose.x, landmarks[263].y - nose.y])
    eye_vec = right_eye - left_eye
    angle = np.arctan2(eye_vec[1], eye_vec[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                 [sin_a,  cos_a]])
    coords[:, :2] = coords[:, :2] @ rotation_matrix.T

    return coords.flatten()

# --- SYSTEM SETTINGS ---
prediction_history = []
stability_threshold = 7
sentence = []
current_char = ""
frames_held = 0
CHAR_LIMIT = 15
last_spoken_index = 0

# Emotion smoothing — same idea as sign smoothing
emotion_history = []
EMOTION_SMOOTH = 10       # Rolling window for emotion stability
current_emotion = "..."   # Default before any face is detected

# --- UI & SCROLLING SETTINGS ---
last_blink_time = time.time()
show_cursor = True
cursor_speed = 0.5
full_transcript = []
scroll_index = 0
MAX_LINES_VISIBLE = 3

# --- LIVE FEED LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

        # Draw hand landmarks only (keeps UI clean)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ============================================================
        # SIGN PREDICTION
        # ============================================================
        sign_keypoints = extract_sign_keypoints(results).reshape(1, -1)

        if np.any(sign_keypoints):
            prediction = sign_model.predict(sign_keypoints)[0]
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

            # Sign label box (top left)
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Progress bar
            bar_width = int(frames_held * (250 / CHAR_LIMIT))
            cv2.rectangle(image, (0, 60), (bar_width, 70), (0, 255, 0), -1)
        else:
            prediction_history = []
            frames_held = 0

        # ============================================================
        # EMOTION PREDICTION
        # ============================================================
        face_keypoints = extract_face_keypoints(results).reshape(1, -1)

        if np.any(face_keypoints):
            emotion_pred = emotion_model.predict(face_keypoints)[0]
            emotion_history.append(emotion_pred)
            emotion_history = emotion_history[-EMOTION_SMOOTH:]
            smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
            current_emotion = emotions[smoothed_emotion]

        # Emotion label box (top right) — colour changes with emotion
        emotion_color = EMOTION_COLORS.get(current_emotion, (200, 200, 200))
        cv2.rectangle(image, (390, 0), (640, 60), emotion_color, -1)
        cv2.putText(image, f'{current_emotion}', (400, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # ============================================================
        # TEXT TERMINAL
        # ============================================================
        cv2.rectangle(image, (0, 350), (640, 480), (245, 117, 16), -1)

        # Blinking cursor
        if time.time() - last_blink_time > cursor_speed:
            show_cursor = not show_cursor
            last_blink_time = time.time()

        display_text = "".join(sentence)

        # Auto line-wrap (visual only — no speaking)
        if len(display_text) > 25:
            full_transcript.append(display_text)
            sentence = []
            display_text = ""
            last_spoken_index = 0
            scroll_index = max(0, len(full_transcript) - MAX_LINES_VISIBLE)

        # Render history lines
        visible_history = full_transcript[scroll_index: scroll_index + MAX_LINES_VISIBLE]
        for i, line in enumerate(visible_history):
            cv2.putText(image, line, (20, 380 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA)

        # Render active line with cursor
        cursor = "|" if show_cursor else ""
        cv2.putText(image, display_text + cursor, (20, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Scroll indicators
        if scroll_index > 0:
            cv2.putText(image, "^", (620, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(full_transcript) > scroll_index + MAX_LINES_VISIBLE:
            cv2.putText(image, "v", (620, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- SPEAK QUEUED TEXT ---
        process_speech()

        cv2.imshow('Sign + Emotion Recognition', image)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence, full_transcript, scroll_index = [], [], 0
            last_spoken_index = 0
        elif key == ord(' '):
            sentence.append(" ")
        elif key == ord('.'):
            sentence.append(".")
            new_text = "".join(sentence[last_spoken_index:])
            speak(new_text)
            last_spoken_index = len(sentence)
        elif key == 8:  # Backspace
            if sentence:
                sentence.pop()
        elif key == 2490368 or key == ord('w'):
            scroll_index = max(0, scroll_index - 1)
        elif key == 2621440 or key == ord('s'):
            scroll_index = min(max(0, len(full_transcript) - MAX_LINES_VISIBLE), scroll_index + 1)

cap.release()
cv2.destroyAllWindows()