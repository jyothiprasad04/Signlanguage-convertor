import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model("lstm_sign_model1.h5")

# Define the actions (same as used in training)
actions = np.array(['Accept', 'Call', 'Copy', 'Find', 'Give', 'Run', 'Shut down', 'Spaghetti', 'Thanks', 'Where'])

# Initialize MediaPipe for pose & hand detection
mp_holistic = mp.solutions.holistic  
mp_drawing = mp.solutions.drawing_utils  

# Extract pose & hand landmarks from frame
def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((33, 3)))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21, 3)))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros((21, 3)))

    return np.concatenate([pose, right_hand, left_hand]).flatten()  # Shape: (225,)

# OpenCV video capture
cap = cv2.VideoCapture(0)

sequence = []
predictions = []
threshold = 0.8  # Confidence threshold

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  
        results = holistic.process(image)  
        image.flags.writeable = True  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  

        # Draw pose & hand landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract and store landmarks
        landmarks = extract_landmarks(results)
        sequence.append(landmarks)

        if len(sequence) == 30:  # Ensure we have a full sequence
            sequence_np = np.expand_dims(np.array(sequence), axis=0)  # Shape: (1, 30, 225)
            prediction = model.predict(sequence_np)[0]
            max_index = np.argmax(prediction)
            confidence = prediction[max_index]

            if confidence > threshold:
                action = actions[max_index]
                predictions.append(action)

            sequence = []  # Reset sequence after prediction

        # Display the predicted action
        if predictions:
            cv2.putText(image, predictions[-1], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Sign Language Recognition", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
