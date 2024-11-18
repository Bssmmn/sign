# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time


# model_dict = pickle.load(open('./model.p' , 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)  # Try index 0 or 1

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0:"A",1:"B"}

# # Give the camera a moment to initialize
# time.sleep(2)





# while True:
#     data_aux=[]


#     ret, frame = cap.read()

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
#                 # Process the image and detect hands
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#          for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
            

#             for hand_landmarks in results.multi_hand_landmarks:
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
#                             y = hand_landmarks.landmark[i].y
#                             data_aux.append(x)  # Append the x-coordinate
#                             data_aux.append(y)  # Append the y-coordinate

#             prediction = model.predict([np.asarray(data_aux)])

#             predicted_charachter = labels_dict[int(prediction[0])]

#             print(predicted_charachter)

#     if not ret:  # If the frame capture fails
#         print("Failed to grab frame.")
#         break

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()









# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Load the trained model from pickle
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize the webcam (index 0 or 1 depending on your system)
# cap = cv2.VideoCapture(0)  # Try index 0 or 1 if 0 doesn't work

# # Setup MediaPipe Hands for hand tracking
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Initialize the hand tracking model
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Define the labels for the characters (extend as needed)
# labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}

# # Allow the camera to initialize for a moment
# time.sleep(2)

# while True:
#     data_aux = []

#     # Capture a frame from the webcam
#     ret, frame = cap.read()

#     # If the frame wasn't captured, exit the loop
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     # Convert the captured frame to RGB for MediaPipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the image and detect hands
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         # Draw the hand landmarks on the frame
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             # Extract hand landmarks and append the x, y coordinates to data_aux
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x)  # Append the x-coordinate
#                     data_aux.append(y)  # Append the y-coordinate

#             # Make a prediction with the model
#             prediction = model.predict([np.asarray(data_aux)])

#             # Ensure the predicted value is within the bounds of labels_dict
#             predicted_value = int(prediction[0])
#             predicted_charachter = labels_dict.get(predicted_value, "Unknown")

#             # Print the predicted character
#             print(f"Predicted values: {prediction}")
#             print(f"Predicted character: {predicted_charachter}")

#     # Show the webcam feed with landmarks (if any)
#     cv2.imshow('frame', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()




import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the video capture (try index 0 or 1 if 2 doesn't work)
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture. Try checking camera permissions or try different indices (0, 1, 2).")
    exit()

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up hand detector
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping from prediction indices to character labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

while True:
    data_aux = []  # List to store relative landmark positions
    x_ = []  # List to store x coordinates of landmarks
    y_ = []  # List to store y coordinates of landmarks

    # Capture a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break  # Break out of the loop if frame capture fails

    # Get the frame dimensions
    H, W, _ = frame.shape

    # Convert the frame to RGB for processing with Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Hand landmarks from Mediapipe
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Append the normalized landmark coordinates to the lists
                x_.append(x)
                y_.append(y)

            # Calculate relative positions of landmarks to the minimum value
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x
                data_aux.append(y - min(y_))  # Normalize y

        # Get bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make prediction using the model
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted character
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and label the prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame with landmarks and prediction
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()
