# this file launches Mediapipe Hands, and performs real-time gesture recognistion
# the predictions of all three models are displayed
import cv2
import mediapipe as mp
import joblib
import numpy as np
import signal
import sys

# Load the classifiers 
svm_clf = joblib.load('svm_gesture_classifier.pkl')
rf_clf = joblib.load('rf_gesture_classifier.pkl')
mlp_clf = joblib.load('mlp_gesture_classifier.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

def extract_landmarks(landmarks):
    return [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]

def signal_handler(sig, frame):
    print("Interrupt received, stopping...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

while cap.isOpened():
    ret, frame = cap.read()

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe operates in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  # Detect hand landmarks

    if results.multi_hand_landmarks:  # If hands are detected
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmark_list = extract_landmarks(landmarks)
            landmark_array = np.array(landmark_list).reshape(1, -1)

            # Predict gesture with all models
            gestures = {
                'SVM': svm_clf.predict(landmark_array)[0],
                'RF': rf_clf.predict(landmark_array)[0],
                'MLP': mlp_clf.predict(landmark_array)[0]
            }
            
            # Display predictions
            y_pos = 30
            for model, gesture in gestures.items():
                cv2.putText(frame, f"{model}: {gesture}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                y_pos += 30

    # Display the title centered and in black
    title = "CV project - Gesture Recognition"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, title, (text_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
