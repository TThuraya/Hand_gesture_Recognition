import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained classifiers
svm_clf = joblib.load('svm_gesture_classifier.pkl')
rf_clf = joblib.load('rf_gesture_classifier.pkl')
mlp_clf = joblib.load('mlp_gesture_classifier.pkl')
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Gesture labels (modify these if you used different labels)
gesture_labels = ['thumbs-up', 'heart', 'open_palm', 'perfect','point_up','point_down','fist']

def extract_landmarks(landmarks):
    return [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]

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

            # Predict gesture with both models
            svm_gesture = svm_clf.predict(landmark_array)[0]
            rf_gesture = rf_clf.predict(landmark_array)[0]
            mlp_gesture = mlp_clf.predict(landmark_array)[0]

            # Display predictions on the frame
            cv2.putText(frame, f"SVM: {svm_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"RF: {rf_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"MLP: {mlp_gesture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
