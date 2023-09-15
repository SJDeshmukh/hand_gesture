import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cam = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8)
max_landmark_count = 21

# Initialize an empty list to store the normalized landmark coordinates
num_frames = 1000
gesture_data = []

# Define the minimum and maximum values for normalization
min_value = 0  # Replace with your desired minimum value
max_value = 1  # Replace with your desired maximum value

for frame_count in range(num_frames):
# while True:
    print(frame_count)
    success, image = cam.read()
    if not success:
        break

    hands, _ = detector.findHands(image)

    if hands:
        for hand_index, hand in enumerate(hands):
            # Get the landmark (keypoint) coordinates for the hand
            hand = hands[0]
            lmList = hand["lmList"]

            if len(lmList) < max_landmark_count:
                pad_length = max_landmark_count - len(lmList)
                lmList += [[0, 0, 0]] * pad_length

            # Normalize the landmark coordinates using Min-Max scaling
            lmList = np.array(lmList)
            lmList = (lmList - lmList.min(axis=0)) / (lmList.max(axis=0) - lmList.min(axis=0))
            lmList = lmList * (max_value - min_value) + min_value

            # Append the list of coordinates to the gesture_data list
            gesture_data.append(lmList.tolist())

            bounding_box = hand["bbox"]
            x, y, w, h = bounding_box

            hand_roi = image[y-10:y+h+10, x-10:x+10+w]  # Adjust cropping parameters if needed
            gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

            _, thresholded_hand = cv2.threshold(gray_hand, 200, 255, cv2.THRESH_BINARY)

            black_background = np.zeros_like(hand_roi)

            final_image = cv2.merge([thresholded_hand, thresholded_hand, thresholded_hand])
            final_image[np.where((final_image == [255, 255, 255]).all(axis=2))] = [255, 255, 255]
            final_image[np.where((final_image == [0, 0, 0]).all(axis=2))] = black_background[np.where((final_image == [0, 0, 0]).all(axis=2))]

            cv2.namedWindow(f'Hand {hand_index + 1} Thresholded', cv2.WINDOW_NORMAL)
            cv2.imshow(f'Hand {hand_index + 1} Thresholded', final_image)
    else:
        cv2.destroyAllWindows()

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

# Convert the landmarks_data list to a NumPy array
landmarks_data_array = np.array(gesture_data)

# Save the NumPy array to a binary file
np.save("gestures/hand_landmarks_thumps_up.npy", landmarks_data_array)
print("Saved")
