import numpy as np

X = np.load("X.npy")  # Shape: (500, 30, 225)
y = np.load("y.npy")  # Shape: (500,)

print(f"Dataset: {X.shape}, Labels: {y.shape}")
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# One-hot encode labels
y = to_categorical(y, num_classes=10)  # 10 actions

# Split into train/test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 225)),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
