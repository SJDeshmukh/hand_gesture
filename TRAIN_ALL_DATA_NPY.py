import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Specify the folder where your gesture data is stored
data_folder = "gestures/"

# Initialize empty lists to store data and labels
X_data = []
y_data = []

# Iterate through files in the folder
for filename in os.listdir(data_folder):
    if filename.endswith(".npy"):
        # Use the filename (excluding the extension) as the label
        label = os.path.splitext(filename)[0]
        
        # Load the gesture data
        gesture_data = np.load(os.path.join(data_folder, filename))
        
        # Append data and label to the lists
        X_data.append(gesture_data)
        y_data.extend([label] * len(gesture_data))

# Convert lists to NumPy arrays if there's valid data
if X_data and y_data:
    X = np.vstack(X_data)
    y = np.array(y_data)
else:
    # Handle the case where no valid data was found
    X = np.empty((0, 21, 3))  # Adjust shape to match your data
    y = np.array([])

# Create a label encoder to convert string labels to integers
label_encoder = LabelEncoder()

# Encode the string labels to integers
y_encoded = label_encoder.fit_transform(y)

# Determine the number of classes
num_classes = len(np.unique(y_encoded))

# Reshape the input data to (num_samples, 21 * 3)
X = X.reshape(X.shape[0], -1)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and compile your model (adjust the architecture as needed)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(21 * 3,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save("GESTURE_recognition_model1.h5")
print("Done Training")
