# Import packages
import cv2
import mediapipe as mp

# Build Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    try:
        image.flags.writeable = False 
    except Exception:
        pass 
    results = model.process(image) 
    try:
        image.flags.writeable = True
    except Exception:
        pass
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results
  
def draw_styled_landmarks(image, results):
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
          image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Draw right hand connections  
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
    
def check_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False
        
    hand = hand_landmarks.landmark
    
    # Same logic as before
    thumb_is_up = hand[4].y < hand[2].y
    
    finger_tips = [8, 12, 16, 20]
    finger_knuckles = [6, 10, 14, 18]
    
    others_are_closed = all(hand[tip].y > hand[knuckle].y 
                            for tip, knuckle in zip(finger_tips, finger_knuckles))
                            
    return thumb_is_up and others_are_closed

def check_peace_sign(hand_landmarks):
    if not hand_landmarks:
        return False
        
    hand = hand_landmarks.landmark
    
    # 1. Fingers that MUST be UP (Index and Middle)
    # Tip Y must be smaller than the knuckle Y
    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    
    # 2. Fingers that MUST be DOWN (Ring and Pinky)
    # Tip Y must be larger than the knuckle Y
    ring_down = hand[16].y > hand[14].y
    pinky_down = hand[20].y > hand[18].y
    
    # 3. Final Check (We ignore the thumb for now to make it easier)
    if index_up and middle_up and ring_down and pinky_down:
        return True
    return False

def check_rock_sign(hand_landmarks):
    if not hand_landmarks:
        return False
        
    hand = hand_landmarks.landmark
    
    # 1. Fingers that MUST be UP (Thumb, Index and Pinky)
    # Tip Y must be smaller than the knuckle Y
    thumb_up = hand[4].y < hand[2].y
    index_up = hand[8].y < hand[6].y
    pinky_up = hand[20].y < hand[18].y
    
    # 2. Fingers that MUST be DOWN (Ring and Middle)
    # Tip Y must be larger than the knuckle Y
    ring_down = hand[16].y > hand[14].y
    middle_down = hand[12].y > hand[10].y
    
    # 3. Final Check
    if index_up and middle_down and ring_down and pinky_up and thumb_up:
        return True
    return False

# Main function
cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # GESTURE LOGIC START
        
        # --- Right Hand Checks ---
        if check_thumbs_up(results.right_hand_landmarks):
            cv2.putText(image, 'R: THUMBS UP', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if check_peace_sign(results.right_hand_landmarks):
            cv2.putText(image, 'R: PEACE SIGN', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if check_rock_sign(results.right_hand_landmarks):
            cv2.putText(image, 'R: ROCK ON SIGN', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Left Hand Checks ---
        if check_thumbs_up(results.left_hand_landmarks):
            cv2.putText(image, 'L: THUMBS UP', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if check_peace_sign(results.left_hand_landmarks):
            cv2.putText(image, 'L: PEACE SIGN', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if check_rock_sign(results.left_hand_landmarks):
            cv2.putText(image, 'L: ROCK ON SIGN', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # GESTURE LOGIC END
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()