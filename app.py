import cv2
import mediapipe as mp
import math # Needed for distance calculation

# Build Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False 
    results = model.process(image) 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results
  
def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

# --- GESTURE FUNCTIONS ---
def check_thumbs_up(hand_landmarks):
    if not hand_landmarks: return False
    hand = hand_landmarks.landmark
    return hand[4].y < hand[2].y and all(hand[t].y > hand[k].y for t,k in zip([8,12,16,20], [6,10,14,18]))

def check_peace_sign(hand_landmarks):
    if not hand_landmarks: return False
    hand = hand_landmarks.landmark
    return (hand[8].y < hand[6].y and hand[12].y < hand[10].y and 
            hand[16].y > hand[14].y and hand[20].y > hand[18].y)

def check_rock_sign(hand_landmarks):
    if not hand_landmarks: return False
    hand = hand_landmarks.landmark
    return (hand[4].y < hand[2].y and hand[8].y < hand[6].y and hand[20].y < hand[18].y and
            hand[16].y > hand[14].y and hand[12].y > hand[10].y)

def check_stop_sign(hand_landmarks):
    if not hand_landmarks: return False
    hand = hand_landmarks.landmark
    fingers_up = all(hand[tip].y < hand[knuckle].y 
                     for tip, knuckle in zip([8, 12, 16, 20], [6, 10, 14, 18]))
    thumb_up = hand[4].y < hand[2].y
    return fingers_up and thumb_up

def check_heart_gesture(results):
    if not (results.left_hand_landmarks and results.right_hand_landmarks): return False
    lh, rh = results.left_hand_landmarks.landmark, results.right_hand_landmarks.landmark
    
    # Check tips distance + index pointing down + middle fingers closed
    tips_close = math.dist((lh[4].x, lh[4].y), (rh[4].x, rh[4].y)) < 0.1 and \
                 math.dist((lh[8].x, lh[8].y), (rh[8].x, rh[8].y)) < 0.1
    
    return tips_close and lh[8].y > lh[6].y and rh[8].y > rh[6].y and lh[12].y > lh[10].y

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # 1. Determine Individual Hand States
        right_msg = ""
        right_stop = False
        if results.right_hand_landmarks:
            right_stop = check_stop_sign(results.right_hand_landmarks)
            if check_thumbs_up(results.right_hand_landmarks): right_msg = "THUMBS UP"
            elif check_peace_sign(results.right_hand_landmarks): right_msg = "PEACE"
            elif check_rock_sign(results.right_hand_landmarks): right_msg = "ROCK ON"
            elif right_stop: right_msg = "HI / STOP"

        left_msg = ""
        left_stop = False
        if results.left_hand_landmarks:
            left_stop = check_stop_sign(results.left_hand_landmarks)
            if check_thumbs_up(results.left_hand_landmarks): left_msg = "THUMBS UP"
            elif check_peace_sign(results.left_hand_landmarks): left_msg = "PEACE"
            elif check_rock_sign(results.left_hand_landmarks): left_msg = "ROCK ON"
            elif left_stop: left_msg = "HI / STOP"

        # 2. INTERACTIVE BANNER LOGIC (Global Gestures)
        banner_color = (0, 0, 0)
        central_msg = ""
        
        if check_heart_gesture(results):
            banner_color = (147, 20, 255) # Pink
            central_msg = "<3 HEART <3"
        elif right_stop and left_stop:
            banner_color = (0, 0, 200) # Red
            central_msg = "!!! STOP !!!"
        elif right_stop or left_stop:
            banner_color = (0, 150, 0) # Green

        # Draw the Banner
        cv2.rectangle(image, (0,0), (width, 60), banner_color, -1)

        # 3. Display Messages
        if central_msg: # If a global gesture (Heart/Stop) is active
             cv2.putText(image, central_msg, (600, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        else: # Show individual hand messages
            cv2.putText(image, f"Right: {right_msg}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Left: {left_msg}", (600, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()