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

# ============================================================
# TTS SETUP
# ============================================================
speech_queue = queue.Queue()

def speak(text):
    clean = text.strip(" .")
    if clean:
        speech_queue.put(clean)

def process_speech():
    if not speech_queue.empty():
        text_to_say = speech_queue.get()
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text_to_say)
        engine.runAndWait()
        engine.stop()

# ============================================================
# LOAD MODELS
# ============================================================
with open('sign_model.p', 'rb') as f:
    sign_dict = pickle.load(f)
sign_model = sign_dict['model']
actions = sign_dict['actions']

with open('emotion_model.p', 'rb') as f:
    emotion_dict = pickle.load(f)
emotion_model = emotion_dict['model']
emotions = emotion_dict['emotions']

# ============================================================
# UI DESIGN CONSTANTS
# Professional dark theme — all colours in BGR
# ============================================================
# Panel colours
C_PANEL_BG      = (30, 30, 30)       # Dark charcoal — main bottom panel
C_PANEL_HEADER  = (20, 20, 20)       # Slightly darker — top strip
C_DIVIDER       = (60, 60, 60)       # Subtle divider line

# Text colours
C_TEXT_ACTIVE   = (255, 255, 255)    # Pure white — active typing line
C_TEXT_HISTORY  = (160, 160, 160)    # Muted grey — history lines
C_TEXT_LABEL    = (200, 200, 200)    # Light grey — UI labels
C_ACCENT        = (0, 200, 120)      # Green accent — progress bar, cursor

# Sign box
C_SIGN_BG       = (45, 45, 45)       # Dark box background
C_SIGN_TEXT     = (255, 255, 255)    # White text

# Emotion colours (BGR) — used as badge background
EMOTION_COLORS = {
    'Happy':     (32, 165, 218),     # Warm amber
    'Sad':       (180, 80,  40),     # Steel blue
    'Angry':     (40,  40, 200),     # Strong red
    'Neutral':   (90,  90,  90),     # Neutral grey
    'Surprised': (160, 120,  0),     # Teal
}

# Emotion emoji (drawn as text — OpenCV doesn't support real emoji)
EMOTION_ICONS = {
    'Happy':     ':)',
    'Sad':       ':(',
    'Angry':     '>:(',
    'Neutral':   ':-|',
    'Surprised': ':O',
}

# Layout constants
W, H           = 640, 480
PANEL_TOP      = 340             # Where the bottom panel starts
HEADER_H       = 52              # Height of top header strip
SIGN_BOX_W     = 220             # Width of sign detection box
EMOTION_BOX_W  = 210             # Width of emotion badge
PROGRESS_H     = 6               # Thickness of progress bar

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
    if not results.face_landmarks:
        return np.zeros(468 * 3)
    landmarks = results.face_landmarks.landmark
    nose = landmarks[1]
    coords = np.array([[lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]
                       for lm in landmarks])
    left_eye  = np.array([landmarks[33].x  - nose.x, landmarks[33].y  - nose.y])
    right_eye = np.array([landmarks[263].x - nose.x, landmarks[263].y - nose.y])
    eye_vec   = right_eye - left_eye
    angle     = np.arctan2(eye_vec[1], eye_vec[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords[:, :2] = coords[:, :2] @ rot.T
    eye_dist = np.linalg.norm(eye_vec)
    if eye_dist > 0:
        coords /= eye_dist
    return coords.flatten()

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draws a rectangle with slightly rounded corners using circles at corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def overlay_panel(img, y_start, alpha=0.85):
    """Blends a dark panel over the camera feed for the text terminal area."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_start), (W, H), C_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ============================================================
# STATE VARIABLES
# ============================================================
prediction_history = []
stability_threshold = 7
sentence      = []
current_char  = ""
frames_held   = 0
CHAR_LIMIT    = 15
last_spoken_index = 0

emotion_history = []
EMOTION_SMOOTH  = 10
current_emotion = "Neutral"

last_blink_time = time.time()
show_cursor     = True
cursor_speed    = 0.5
full_transcript = []
scroll_index    = 0
MAX_LINES_VISIBLE = 3

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

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

        # Draw hand landmarks — white dots, clean style
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1))

        # ============================================================
        # SIGN PREDICTION
        # ============================================================
        sign_kp = extract_sign_keypoints(results).reshape(1, -1)
        hand_detected = np.any(sign_kp)

        if hand_detected:
            prediction = sign_model.predict(sign_kp)[0]
            prediction_history.append(prediction)
            prediction_history = prediction_history[-stability_threshold:]
            most_common  = Counter(prediction_history).most_common(1)[0][0]
            predicted_char = actions[most_common]

            if predicted_char == current_char:
                frames_held += 1
            else:
                current_char = predicted_char
                frames_held  = 0

            if frames_held == CHAR_LIMIT:
                sentence.append(predicted_char)
                frames_held = 0
        else:
            prediction_history = []
            frames_held = 0
            predicted_char = current_char  # Hold last char for display

        # ============================================================
        # EMOTION PREDICTION
        # ============================================================
        face_kp = extract_face_keypoints(results).reshape(1, -1)
        if np.any(face_kp):
            emotion_pred = emotion_model.predict(face_kp)[0]
            emotion_history.append(emotion_pred)
            emotion_history = emotion_history[-EMOTION_SMOOTH:]
            current_emotion = emotions[Counter(emotion_history).most_common(1)[0][0]]

        # ============================================================
        # UI — TOP HEADER STRIP (dark background across full width)
        # ============================================================
        # Semi-transparent dark strip at top
        top_overlay = image.copy()
        cv2.rectangle(top_overlay, (0, 0), (W, HEADER_H), C_PANEL_HEADER, -1)
        cv2.addWeighted(top_overlay, 0.75, image, 0.25, 0, image)

        # App title — centred, subtle
        cv2.putText(image, 'SignSpeak', (W//2 - 55, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA)

        # --- SIGN BOX (top left) ---
        if hand_detected:
            sign_label = predicted_char
            sign_box_color = (50, 50, 50)
        else:
            sign_label = '—'
            sign_box_color = (35, 35, 35)

        draw_rounded_rect(image, (10, 8), (10 + SIGN_BOX_W, HEADER_H - 8),
                          sign_box_color, radius=6)
        cv2.putText(image, 'SIGN', (22, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(image, sign_label, (70, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_SIGN_TEXT, 2, cv2.LINE_AA)

        # Progress bar — slim green bar below sign box
        if hand_detected and frames_held > 0:
            bar_w = int(frames_held * (SIGN_BOX_W / CHAR_LIMIT))
            cv2.rectangle(image, (10, HEADER_H - 6), (10 + SIGN_BOX_W, HEADER_H - 6 + PROGRESS_H),
                          (40, 40, 40), -1)
            cv2.rectangle(image, (10, HEADER_H - 6), (10 + bar_w, HEADER_H - 6 + PROGRESS_H),
                          C_ACCENT, -1)

        # --- EMOTION BADGE (top right) ---
        emotion_color = EMOTION_COLORS.get(current_emotion, (80, 80, 80))
        icon = EMOTION_ICONS.get(current_emotion, '')
        ex1 = W - EMOTION_BOX_W - 10
        draw_rounded_rect(image, (ex1, 8), (W - 10, HEADER_H - 8),
                          emotion_color, radius=6)
        cv2.putText(image, 'MOOD', (ex1 + 10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f'{icon}  {current_emotion}', (ex1 + 10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # ============================================================
        # UI — BOTTOM TEXT PANEL
        # ============================================================
        overlay_panel(image, PANEL_TOP, alpha=0.92)

        # Thin accent line at top of panel
        cv2.line(image, (0, PANEL_TOP), (W, PANEL_TOP), C_ACCENT, 1)

        # Blinking cursor
        if time.time() - last_blink_time > cursor_speed:
            show_cursor = not show_cursor
            last_blink_time = time.time()

        display_text = "".join(sentence)

        # Auto line-wrap
        if len(display_text) > 25:
            full_transcript.append(display_text)
            sentence      = []
            display_text  = ""
            last_spoken_index = 0
            scroll_index  = max(0, len(full_transcript) - MAX_LINES_VISIBLE)

        # History lines
        visible_history = full_transcript[scroll_index: scroll_index + MAX_LINES_VISIBLE]
        for i, line in enumerate(visible_history):
            cv2.putText(image, line, (18, PANEL_TOP + 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT_HISTORY, 1, cv2.LINE_AA)

        # Divider between history and active line
        divider_y = H - 52
        cv2.line(image, (18, divider_y), (W - 18, divider_y), C_DIVIDER, 1)

        # Active typing line
        cursor_char = "|" if show_cursor else " "
        cv2.putText(image, display_text + cursor_char, (18, H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TEXT_ACTIVE, 2, cv2.LINE_AA)

        # Scroll indicators
        if scroll_index > 0:
            cv2.putText(image, "▲", (W - 20, PANEL_TOP + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT_LABEL, 1)
        if len(full_transcript) > scroll_index + MAX_LINES_VISIBLE:
            cv2.putText(image, "▼", (W - 20, divider_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT_LABEL, 1)

        # Controls hint — bottom right corner
        cv2.putText(image, "SPC=space  .=speak  DEL=back  C=clear",
                    (18, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (70, 70, 70), 1, cv2.LINE_AA)

        # ============================================================
        # SPEAK & DISPLAY
        # ============================================================
        process_speech()
        cv2.imshow('SignSpeak', image)

        # ============================================================
        # KEYBOARD CONTROLS
        # ============================================================
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
            speak("".join(sentence[last_spoken_index:]))
            last_spoken_index = len(sentence)
        elif key == 8:
            if sentence:
                sentence.pop()
        elif key == 2490368 or key == ord('w'):
            scroll_index = max(0, scroll_index - 1)
        elif key == 2621440 or key == ord('s'):
            scroll_index = min(max(0, len(full_transcript) - MAX_LINES_VISIBLE), scroll_index + 1)

cap.release()
cv2.destroyAllWindows()