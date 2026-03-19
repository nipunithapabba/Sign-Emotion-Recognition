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
        # Check Right Hand
        if check_thumbs_up(results.right_hand_landmarks):
            cv2.putText(image, 'RIGHT THUMBS UP', (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Check Left Hand
        if check_thumbs_up(results.left_hand_landmarks):
            cv2.putText(image, 'LEFT THUMBS UP', (50, 200), # Notice Y is 200 so they don't overlap
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # GESTURE LOGIC END
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()