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

# Load emoji images
emoji_images = {
    'thumbs-up': cv2.imread('thumbs_up_emoji.png', cv2.IMREAD_UNCHANGED),
    'heart': cv2.imread('heart_emoji.png', cv2.IMREAD_UNCHANGED),
    'open_palm': cv2.imread('open_palm_emoji.png', cv2.IMREAD_UNCHANGED),
    'perfect': cv2.imread('perfect_emoji.png', cv2.IMREAD_UNCHANGED),
    'point_up': cv2.imread('point_up_emoji.png', cv2.IMREAD_UNCHANGED),
    'point_down': cv2.imread('point_down_emoji.png', cv2.IMREAD_UNCHANGED),
    'fist': cv2.imread('fist_emoji.png', cv2.IMREAD_UNCHANGED)
}

def extract_landmarks(landmarks):
    return [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    x and y with an alpha mask."""
    img_overlay = cv2.resize(img_overlay, (50, 50))
    b, g, r, a = cv2.split(img_overlay)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    img_roi = img[y:y+50, x:x+50]
    img_roi = cv2.add(img_roi, overlay_color, mask=mask)
    img[y:y+50, x:x+50] = img_roi

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
            
            # Display predictions and emojis
            y_pos = 30
            for model, gesture in gestures.items():
                cv2.putText(frame, f"{model}: {gesture}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                emoji_img = emoji_images.get(gesture)
                if emoji_img is not None:
                    overlay_image_alpha(frame, emoji_img, 200, y_pos - 25, emoji_img[:, :, 3])  # Adjust position as needed
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
