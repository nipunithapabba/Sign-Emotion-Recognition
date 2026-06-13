import sys
sys.modules['tensorflow'] = None
import cv2
import numpy as np
import os
import mediapipe as mp

# ============================================================
# SETTINGS
# ============================================================
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised']
NO_SAMPLES = 100      # more data = better accuracy
DATA_PATH = 'Emotion_Data'

# ============================================================
# WIPE OLD DATA & RECREATE FOLDERS
# (We're recollecting everything fresh with better normalization)
# ============================================================
import shutil
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
    print("Old Emotion_Data wiped — collecting fresh data.")

for emotion in EMOTIONS:
    os.makedirs(os.path.join(DATA_PATH, emotion), exist_ok=True)

# ============================================================
# MEDIAPIPE
# ============================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_face_keypoints(results):
    """
    Extracts face landmarks with TWO layers of normalization:

    1. TRANSLATION — subtract nose tip so position in frame doesn't matter
    2. ROTATION — rotate all points to correct for head tilt
       We use the line between the LEFT EYE (landmark 33) and RIGHT EYE (landmark 263)
       to calculate the tilt angle, then rotate everything so that line is always horizontal.
       This means tilting your head up/down no longer changes the feature values.
    """
    if not results.face_landmarks:
        return np.zeros(468 * 3)

    landmarks = results.face_landmarks.landmark

    # Step 1: Translate — nose tip as origin
    nose = landmarks[1]
    coords = np.array([[lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]
                       for lm in landmarks])

    # Step 2: Rotate — correct for head tilt using eye line angle
    left_eye  = np.array([landmarks[33].x  - nose.x, landmarks[33].y  - nose.y])
    right_eye = np.array([landmarks[263].x - nose.x, landmarks[263].y - nose.y])

    # Angle of the eye line relative to horizontal
    eye_vec = right_eye - left_eye
    angle = np.arctan2(eye_vec[1], eye_vec[0])  # Radians

    # 2D rotation matrix (we only rotate x,y — z depth stays as is)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                 [sin_a,  cos_a]])

    # Apply rotation to x,y coordinates only
    xy_rotated = coords[:, :2] @ rotation_matrix.T
    coords[:, :2] = xy_rotated

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
        print(f"\n>>> GET READY FOR: {emotion.upper()} (100 samples)")

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

            # Draw face mesh
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1)
                )

            # --- UI ---
            cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(image, f'{emotion.upper()}  |  Sample: {sample + 1}/{NO_SAMPLES}',
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Instructions based on progress
            # First third: straight face
            # Second third: slight tilt up
            # Last third: slight tilt down
            # This teaches the model that emotion ≠ tilt
            if sample < 34:
                tip = "Look straight at the camera"
            elif sample < 67:
                tip = "Slightly tilt head UP (keep expression!)"
            else:
                tip = "Slightly tilt head DOWN (keep expression!)"

            cv2.putText(image, tip, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Progress bar for current emotion
            bar_w = int((sample / NO_SAMPLES) * 620)
            cv2.rectangle(image, (10, 90), (630, 105), (50, 50, 50), -1)
            cv2.rectangle(image, (10, 90), (10 + bar_w, 105), (0, 200, 0), -1)

            # Countdown before first sample
            if sample == 0:
                for countdown in range(10, 0, -1):
                    ret2, frame2 = cap.read()
                    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    img2.flags.writeable = False
                    results2 = holistic.process(img2)
                    img2.flags.writeable = True
                    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img2, (0, 0), (640, 50), (0, 0, 0), -1)
                    cv2.putText(img2, f'{emotion.upper()}  |  Sample: 1/{NO_SAMPLES}',
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img2, f'Make your {emotion} face in {countdown}...',
                                (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.imshow('Emotion Data Collection', img2)
                    cv2.waitKey(1000)

            cv2.imshow('Emotion Data Collection', image)
            cv2.waitKey(80)  # Slightly faster than before — 80ms between samples

            # Only save if face is actually detected
            keypoints = extract_face_keypoints(results)
            if np.any(keypoints):
                save_path = os.path.join(DATA_PATH, emotion, f'{sample}.npy')
                np.save(save_path, keypoints)
                sample += 1  # Only increment if we got a valid frame
            else:
                cv2.putText(image, '⚠ No face detected — move closer!', (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        print(f"✅ {emotion} — {NO_SAMPLES} samples saved!")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All done! Run train_emotion_model.py next.")