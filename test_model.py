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
# We use a queue so speech requests don't interrupt the camera loop.
# On Windows, pyttsx3 gets "stuck" if you reuse the engine after runAndWait().
# Fix: reinitialize a fresh engine each time we speak. Reliable, if slightly slower.
speech_queue = queue.Queue()

def speak(text):
    """Add text to the speech queue (strips dots/spaces so it sounds natural)."""
    clean = text.strip(" .")
    if clean:
        speech_queue.put(clean)

def process_speech():
    """
    Called once per frame. If something is waiting in the queue, speak it.
    Reinitializes the engine fresh each time — this is the Windows-safe fix.
    """
    if not speech_queue.empty():
        text_to_say = speech_queue.get()
        engine = pyttsx3.init()           # Fresh engine every time
        engine.setProperty('rate', 160)   # Comfortable speaking speed
        engine.say(text_to_say)
        engine.runAndWait()
        engine.stop()                     # Fully release the engine

# --- LOAD THE BRAIN ---
with open('sign_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
actions = model_dict['actions']

# --- MEDIAPIPE SETUP ---
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
        rh = np.zeros(21 * 3)
    return rh

# --- SYSTEM SETTINGS ---
prediction_history = []
stability_threshold = 7   # Reduced from 10 — reacts to your sign a bit faster
sentence = []
current_char = ""
frames_held = 0
CHAR_LIMIT = 15  # Reduced from 25 — shorter hold time before letter commits
last_spoken_index = 0  # Tracks where in sentence[] we last spoke up to

# --- UI & SCROLLING SETTINGS ---
last_blink_time = time.time()
show_cursor = True
cursor_speed = 0.5
full_transcript = []
scroll_index = 0
MAX_LINES_VISIBLE = 3

# --- LIVE FEED LOOP ---
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

            # Top recognition bar
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, f'SIGN: {predicted_char}', (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Progress bar
            bar_width = int(frames_held * (250 / CHAR_LIMIT))
            cv2.rectangle(image, (0, 60), (bar_width, 70), (0, 255, 0), -1)
        else:
            prediction_history = []
            frames_held = 0

        # --- TEXT TERMINAL ---
        cv2.rectangle(image, (0, 350), (640, 480), (245, 117, 16), -1)

        # Blinking cursor
        if time.time() - last_blink_time > cursor_speed:
            show_cursor = not show_cursor
            last_blink_time = time.time()

        display_text = "".join(sentence)

        # Auto line-wrap trigger (visual only — no speaking here)
        if len(display_text) > 25:
            full_transcript.append(display_text)
            sentence = []
            display_text = ""
            last_spoken_index = 0  # Reset bookmark since buffer cleared
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

        # --- SPEAK QUEUED TEXT (Windows-safe, runs on main thread) ---
        process_speech()

        cv2.imshow('Sign Language Sentence Builder', image)

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
            # Only speak the NEW part since the last period, not the whole buffer
            new_text = "".join(sentence[last_spoken_index:])
            speak(new_text)
            last_spoken_index = len(sentence)  # Move marker forward
        elif key == 8:  # Backspace
            if sentence:
                sentence.pop()
        elif key == 2490368 or key == ord('w'):
            scroll_index = max(0, scroll_index - 1)
        elif key == 2621440 or key == ord('s'):
            scroll_index = min(max(0, len(full_transcript) - MAX_LINES_VISIBLE), scroll_index + 1)

cap.release()
cv2.destroyAllWindows()