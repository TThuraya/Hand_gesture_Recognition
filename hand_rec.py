from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import joblib
import numpy as np

app = Flask(__name__)

# Load the classifiers for gesture recognition
svm_clf = joblib.load('/Users/shahadaleissa/Hand_gesture_Recognition/training/mlp_gesture_classifier.pkl')
rf_clf = joblib.load('/Users/shahadaleissa/Hand_gesture_Recognition/training/rf_gesture_classifier.pkl')
mlp_clf = joblib.load('/Users/shahadaleissa/Hand_gesture_Recognition/training/mlp_gesture_classifier.pkl')

# Initialize MediaPipe Hands and Holistic models
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Video capture object
cap = cv2.VideoCapture(0)
streaming = True  # Flag to control streaming
mode = 'gesture'  # Default mode

def extract_landmarks(landmarks):
    return [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]

def generate_frames():
    global streaming, mode
    while streaming:
        success, frame = cap.read()
        if not success:
            break

        # Mirror the image
        frame = cv2.flip(frame, 1)

        if mode == 'gesture':
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
        elif mode == 'face_mesh':
            # Convert to RGB (MediaPipe operates in RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)  # Detect holistic features

            # Draw face mesh and hand landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display the title centered and in black
        title = "CV project - Gesture Recognition" if mode == 'gesture' else "CV project - Face Mesh"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global streaming
    streaming = True
    cap.open(0)  # Ensure the capture is open
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/stop_feed')
def stop_feed():
    global streaming
    streaming = False
    cap.release()
    cv2.destroyAllWindows()
    return jsonify(status="Stream stopped")

@app.route('/set_mode/<new_mode>')
def set_mode(new_mode):
    global mode
    mode = new_mode
    return jsonify(status=f"Mode set to {mode}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)