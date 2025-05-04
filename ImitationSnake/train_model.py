import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === Step 1: Load data ===
CSV_PATH = 'filtered_snake_data.csv'
df = pd.read_csv(CSV_PATH)

# === Step 2: Preprocess ===

direction_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
df['Direction_Label'] = df['Direction'].map(direction_map)

# Parse board into numpy arrays
def parse_board(board_str):
    board_str = board_str.replace('[', '').replace(']', '')
    board_nums = np.fromstring(board_str, sep=' ')
    if board_nums.size != 256:
        raise ValueError(f"Board parsing error. Got {board_nums.size} elements instead of 256.")
    return board_nums.reshape((16, 16))

df['Parsed_Board'] = df['Board State'].apply(parse_board)

# Prepare input/output
X = np.stack(df['Parsed_Board'].values).astype('float32') 
X = X.reshape(-1, 16, 16, 1)
y = df['Direction_Label'].values
y = to_categorical(y, num_classes=4)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# === Step 3: Build CNN model ===
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(4, activation='softmax') # one for each direction, hope it doesn't output backwards
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 4: Train ===
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# === Step 5: Save the model ===
model.save("snake_cnn_model.keras")
print("Model trained and saved to snake_cnn_model.keras")
