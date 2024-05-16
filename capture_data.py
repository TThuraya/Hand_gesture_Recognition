
# This fioe utlizes Mediapipe hands to capture a dataset of our desiered hand gestures 
#it launches The Mediapipe Hands, detects the hands from the camera and iputs the corresponding label to the gesture
# the collected database 'gesture_data_org' was captured with 591 records and 6 classes (gestures)
# the six gestures we chose were 'thumbs_up', 'point_up', 'point_down', 'open_palm', 'heart', 'perfect'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initializing MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Store labels for each gesture, we can add new gestures anytime, this can be expanded in the future to read sign language 
gesture_labels = []  
# Storing landmarks, landmarks are the cordonates of each joint in the hand skelaton offered by Mediapipe
landmarks_list = []

def extract_landmarks(landmarks):
    return [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]

while cap.isOpened():
    ret, frame = cap.read()

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe only operates in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect hand landmarks
    results = hands.process(rgb_frame)  


    if results.multi_hand_landmarks:  # If hands are detected
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and save landmarks with their corresponding gesture label
            gesture = input("Enter gesture label: ")
            gesture_labels.append(gesture)
            landmarks_list.append(extract_landmarks(landmarks))

    cv2.imshow("Gesture Capture", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the landmarks and labels to a CSV file
data = pd.DataFrame(landmarks_list)
data['label'] = gesture_labels
data.to_csv('gesture_data_org.csv', index=False)
