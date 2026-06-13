import sys
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import os
import mediapipe as mp
import shutil

# ============================================================
# SETTINGS — full clean sweep of all 5 emotions
# ============================================================
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised']
NO_SAMPLES = 150
DATA_PATH = 'Emotion_Data'

EMOTION_TIPS = {
    'Happy': [
        "Big genuine smile — cheeks pushed up",
        "Think of something that actually makes you happy",
        "Crow's feet at eyes — real smile not forced",
        "Keep smiling naturally!",
    ],
    'Sad': [
        "Pull mouth corners DOWN firmly",
        "Eyebrows angled inward and upward — puppy face",
        "Slightly droopy eyelids, look a little down",
        "Subtle but held steady — don't let it fade",
    ],
    'Angry': [
        "PULL eyebrows DOWN and TOGETHER hard",
        "Squint slightly, jaw clenched",
        "Stare intensely — like you're really annoyed",
        "Think of something that genuinely frustrates you",
    ],
    'Neutral': [
        "ZERO expression — completely relaxed",
        "Let your jaw go loose, don't clench",
        "Don't smile even slightly",
        "This is your face doing absolutely nothing",
    ],
    'Surprised': [
        "Eyebrows SHOT UP as high as they go",
        "Eyes wide open — really wide",
        "Mouth slightly open in an O shape",
        "Like someone just surprised you!",
    ],
}

# Wipe everything and start fresh
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
for emotion in EMOTIONS:
    os.makedirs(os.path.join(DATA_PATH, emotion))
print("Fresh start — all old data cleared.\n")

# ============================================================
# MEDIAPIPE
# ============================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_face_keypoints(results):
    """Translation + Rotation + Scale normalization."""
    if not results.face_landmarks:
        return np.zeros(468 * 3)
    landmarks = results.face_landmarks.landmark
    nose = landmarks[1]
    coords = np.array([[lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]
                       for lm in landmarks])
    left_eye  = np.array([landmarks[33].x  - nose.x, landmarks[33].y  - nose.y])
    right_eye = np.array([landmarks[263].x - nose.x, landmarks[263].y - nose.y])
    eye_vec = right_eye - left_eye
    angle = np.arctan2(eye_vec[1], eye_vec[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords[:, :2] = coords[:, :2] @ rot.T
    eye_distance = np.linalg.norm(eye_vec)
    if eye_distance > 0:
        coords /= eye_distance
    return coords.flatten()

# ============================================================
# COLLECTION LOOP
# ============================================================
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for emotion in EMOTIONS:
        tips = EMOTION_TIPS[emotion]
        tip_change_every = NO_SAMPLES // len(tips)
        print(f"\n>>> COLLECTING: {emotion.upper()} ({NO_SAMPLES} samples)")

        sample = 0
        while sample < NO_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1)
                )

            # UI
            cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(image, f'{emotion.upper()}  |  {sample + 1}/{NO_SAMPLES}',
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            tip_index = min(sample // tip_change_every, len(tips) - 1)
            cv2.rectangle(image, (0, 52), (640, 82), (20, 20, 20), -1)
            cv2.putText(image, f'TIP: {tips[tip_index]}',
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            if sample < 50:
                head_tip = "Head STRAIGHT"
            elif sample < 100:
                head_tip = "Slight tilt UP"
            else:
                head_tip = "Slight tilt DOWN"
            cv2.putText(image, head_tip, (500, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

            bar_w = int((sample / NO_SAMPLES) * 620)
            cv2.rectangle(image, (10, 88), (630, 103), (50, 50, 50), -1)
            cv2.rectangle(image, (10, 88), (10 + bar_w, 103), (0, 200, 0), -1)

            # Countdown before first sample of each emotion
            if sample == 0:
                for countdown in range(5, 0, -1):
                    ret2, frame2 = cap.read()
                    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    img2.flags.writeable = False
                    holistic.process(img2)
                    img2.flags.writeable = True
                    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img2, (0, 0), (640, 50), (0, 0, 0), -1)
                    cv2.putText(img2, f'{emotion.upper()} — get ready!',
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img2, tips[0],
                                (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(img2, f'Starting in {countdown}...',
                                (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.imshow('Emotion Data Collection', img2)
                    cv2.waitKey(1000)

            cv2.imshow('Emotion Data Collection', image)
            cv2.waitKey(80)

            keypoints = extract_face_keypoints(results)
            if np.any(keypoints):
                np.save(os.path.join(DATA_PATH, emotion, f'{sample}.npy'), keypoints)
                sample += 1
            else:
                cv2.putText(image, 'No face detected — move closer!',
                            (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Emotion Data Collection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        print(f"✅ {emotion} — {NO_SAMPLES} samples saved!")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All 5 emotions collected! Run train_emotion_model.py now.")