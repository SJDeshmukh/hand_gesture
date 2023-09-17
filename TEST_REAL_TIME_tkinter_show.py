import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import Canvas, Label
from PIL import Image, ImageTk

# Load the trained model
model = tf.keras.models.load_model("GESTURE_recognition_model1.h5")

# Initialize OpenCV's hand tracking module
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8)

# Create a label encoder to convert integer labels back to string labels
label_encoder = LabelEncoder()

# Inverse transform labels to get human-readable class names
class_names = {
    0: "Unknown",
    1: "Peace",
    2: "Pointer",
    3: "Thumps_Down",
    4: "Thumps_Up"
    # Add the actual class names you used during training
}

# Initialize the tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")

# Create a canvas widget for displaying the camera feed
canvas = Canvas(root, width=640, height=480)
canvas.pack()

# Create a label widget for displaying the predicted gesture
gesture_label = Label(root, text="", font=("Arial", 18))
gesture_label.pack()

# Initialize the camera
cap = cv2.VideoCapture(0)

def update():
    try:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Detect hands in the frame
        hands, _ = detector.findHands(frame)

        if hands:
            # Assuming you are processing the first detected hand
            hand = hands[0]
            lmList = hand["lmList"]

            # Normalize the landmark coordinates using Min-Max scaling
            lmList = np.array(lmList)
            min_value = 0  # Replace with your desired minimum value
            max_value = 1  # Replace with your desired maximum value

            # Calculate the minimum and maximum values for each dimension (x, y, z)
            min_vals = lmList.min(axis=0)
            max_vals = lmList.max(axis=0)

            # Normalize the landmark coordinates
            lmList = (lmList - min_vals) / (max_vals - min_vals)
            lmList = lmList * (max_value - min_value) + min_value

            # Reshape the data to match the input shape of your model
            X_new = lmList.reshape(1, -1)

            # Make a prediction using the trained model
            predictions = model.predict(X_new)
            predicted_class = np.argmax(predictions, axis=1)

            # Get the human-readable class name
            predicted_class_name = class_names.get(predicted_class[0], "Unknown")

            # Update the label widget with the predicted gesture
            gesture_label.config(text=predicted_class_name)

        # Convert the OpenCV frame to a Tkinter-compatible image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, image=img, anchor=tk.NW)
        canvas.img = img  # Keep a reference to prevent image garbage collection

        # Schedule the next update
        root.after(10, update)
    except:
        pass

# Schedule the initial update
update()

# Start the Tkinter main loop
root.mainloop()

# Release the camera when the application exits
cap.release()
cv2.destroyAllWindows()
