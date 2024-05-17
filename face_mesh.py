import cv2
import mediapipe as mp
import signal
import sys

# Initialize mediapipe holistic model and drawing utils
holistic_model = mp.solutions.holistic
drawing_utils = mp.solutions.drawing_utils

# Function to make detections
def detect_mediapipe_features(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make predictions
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion back
    return image, results

# Function to draw styled landmarks
def draw_custom_landmarks(image, results):
    # Draw face connections
    drawing_utils.draw_landmarks(
        image, results.face_landmarks, holistic_model.FACEMESH_TESSELATION,
        drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
  
    # Draw left hand connections
    drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, holistic_model.HAND_CONNECTIONS,
        drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, holistic_model.HAND_CONNECTIONS,
        drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("Interrupt received, stopping...")
    video_capture.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Set up the mediapipe model
with holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while video_capture.isOpened():
        # Read the feed
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Make detections
        frame, detection_results = detect_mediapipe_features(frame, holistic)
        
        # Draw landmarks
        draw_custom_landmarks(frame, detection_results)

        # Show the image
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
