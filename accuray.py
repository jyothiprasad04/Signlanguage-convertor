import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load test dataset
X_test = np.load("X.npy")  # Replace with actual test data path
y_test = np.load("y.npy")  # Replace with actual test labels path

# Load trained LSTM model
model = load_model("lstm_sign_model1.h5")  # Replace with your model file

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")
