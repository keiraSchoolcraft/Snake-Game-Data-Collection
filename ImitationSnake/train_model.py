import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === Step 1: Load data ===
BOARDS_PATH = "processed_snake_data.npz"
board_data = np.load(BOARDS_PATH)

X = board_data['states'].astype(np.float32) / 3.0 # normalize
X = np.transpose(X, (0, 2, 3, 1)) # TF expects channels first

# one hot encode labels
y = to_categorical(board_data['actions'], num_classes=4)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# === Step 2: Build CNN model ===
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(4, activation='softmax') # one for each direction, hope it doesn't output backwards
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 3: Train ===
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# === Step 4: Save the model ===
model.save("snake_cnn_model.keras")
print("Model trained and saved to snake_cnn_model.keras")
