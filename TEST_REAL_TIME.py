import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model("GESTURE_recognition_model1.h5")

# Initialize OpenCV's hand tracking module
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a label encoder to convert integer labels back to string labels
label_encoder = LabelEncoder()

# Inverse transform labels to get human-readable class names
class_names = {
    0: "Unknown",
    1: "Peace",
    2: "Thumps_Down",
    3: "Thumps_Up"
    # Add the actual class names you used during training
}

while True:
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
            
            # Calculate the Euclidean distance between the detected gesture and model predictions
            # euclidean_distance = np.linalg.norm(X_new - predictions)
            # Display the predicted class and Euclidean distance on the frame
            # text = f"Class: {predicted_class_name}, Euclidean Distance: {euclidean_distance:.2f}"
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            # If the Euclidean distance is below a certain threshold, consider it as a match
            # euclidean_threshold = 0.9  # Adjust the threshold as needed
            # if euclidean_distance < euclidean_threshold:
            #     predicted_class_name = "Matched"
            #     cv2.putText(frame, predicted_class_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imshow("Gesture Recognition", frame)
            # else:
            #     predicted_class_name = "UnMatched"
            #     cv2.putText(frame, predicted_class_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imshow("Gesture Recognition", frame)
        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    except:
        pass

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
