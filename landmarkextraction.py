import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define actions (your 10 classes)
actions = ['Accept', 'Call', 'Copy', 'Find', 'Give', 'Run', 'Shut down', 'Spaghetti', 'Thanks', 'Where']

# Fixed sequence length (e.g., 30 frames per video)
SEQUENCE_LENGTH = 30

# Function to extract landmarks
def extract_landmarks(results):
    pose = results.pose_landmarks.landmark if results.pose_landmarks else []
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
    
    # Convert to NumPy array and flatten
    pose = np.array([[lm.x, lm.y, lm.z] for lm in pose]).flatten() if pose else np.zeros(33 * 3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in lh]).flatten() if lh else np.zeros(21 * 3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in rh]).flatten() if rh else np.zeros(21 * 3)
    
    return np.concatenate([pose, lh, rh])

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []
    
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Extract landmarks
            landmarks = extract_landmarks(results)
            landmarks_sequence.append(landmarks)
    
    cap.release()
    
    # Ensure fixed length using padding or interpolation
    if len(landmarks_sequence) < SEQUENCE_LENGTH:
        padding = [np.zeros_like(landmarks_sequence[0])] * (SEQUENCE_LENGTH - len(landmarks_sequence))
        landmarks_sequence.extend(padding)
    else:
        landmarks_sequence = landmarks_sequence[:SEQUENCE_LENGTH]
    
    return np.array(landmarks_sequence)

# Prepare dataset
X, y = [], []

dataset_path = "10signdata"

for action_idx, action in enumerate(actions):
    action_folder = os.path.join(dataset_path, action)
    
    for video_file in tqdm(os.listdir(action_folder), desc=f"Processing {action}"):
        video_path = os.path.join(action_folder, video_file)
        sequence = process_video(video_path)
        
        X.append(sequence)
        y.append(action_idx)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save the dataset
np.save("X.npy", X)
np.save("y.npy", y)

print(f"Dataset saved! Shape: {X.shape}, Labels: {y.shape}")
