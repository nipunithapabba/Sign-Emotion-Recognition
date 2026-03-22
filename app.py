import cv2
import mediapipe as mp

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

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

# Get camera dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # 1. Determine states for each hand
        right_msg = ""
        right_stop = False
        if results.right_hand_landmarks:
            right_stop = check_stop_sign(results.right_hand_landmarks)
            if check_thumbs_up(results.right_hand_landmarks): right_msg = "THUMBS UP"
            elif check_peace_sign(results.right_hand_landmarks): right_msg = "PEACE"
            elif check_rock_sign(results.right_hand_landmarks): right_msg = "ROCK ON"
            elif right_stop: right_msg = "STOP / HI"

        left_msg = ""
        left_stop = False
        if results.left_hand_landmarks:
            left_stop = check_stop_sign(results.left_hand_landmarks)
            if check_thumbs_up(results.left_hand_landmarks): left_msg = "THUMBS UP"
            elif check_peace_sign(results.left_hand_landmarks): left_msg = "PEACE"
            elif check_rock_sign(results.left_hand_landmarks): left_msg = "ROCK ON"
            elif left_stop: left_msg = "STOP / HI"

        # 2. INTERACTIVE BANNER LOGIC
        banner_color = (0, 0, 0) # Default Black
        if right_stop and left_stop:
            banner_color = (0, 0, 200) # Emergency Red
            right_msg = "!!! STOP !!!"
            left_msg = "!!! STOP !!!"
        elif right_stop or left_stop:
            banner_color = (0, 150, 0) # Friendly Green

        # Draw the Banner
        cv2.rectangle(image, (0,0), (width, 60), banner_color, -1)

        # 3. Show messages in the banner
        cv2.putText(image, f"Right: {right_msg}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Using the width variable to position Left message perfectly
        cv2.putText(image, f"Left: {left_msg}", (640, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()