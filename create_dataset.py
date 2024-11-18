# import os
# import pickle
# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING logs


# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []  # Fixed variable name from 'lables' to 'labels'

# # Iterate over directories in DATA_DIR
# for dir_ in os.listdir(DATA_DIR):
#     dir_path = os.path.join(DATA_DIR, dir_)
    
#     # Check if the path is a directory
#     if os.path.isdir(dir_path):
#         for img_path in os.listdir(dir_path):  # Process all images in each directory
#             data_aux = []
#             img = cv2.imread(os.path.join(dir_path, img_path))
            
#             if img is not None:
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
#                 # Process the image and detect hands
#                 results = hands.process(img_rgb)
                
#                 if results.multi_hand_landmarks:  # If hands are detected
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
#                             y = hand_landmarks.landmark[i].y
#                             data_aux.append(x)  # Append the x-coordinate
#                             data_aux.append(y)  # Append the y-coordinate

#                     # Append the landmarks data and the corresponding label
#                     data.append(data_aux)
#                     labels.append(dir_)

#     # Display image (optional, commented for now)
#     # plt.figure()
#     # plt.imshow(img_rgb)
#     # plt.title(f"Hand landmarks in {img_path}")
#     # plt.show()

# # Uncomment this part to display the data
# # print("Data:", data)
# # print("Labels:", labels)

# # Optionally save the data to a file for later use, like CSV or numpy file
# # Example:
# # import numpy as np
# # np.savetxt("hand_landmarks_data.csv", data, delimiter=",")
# # np.savetxt("hand_labels.csv", labels, delimiter=",")

# f = open("data.pickle","wb")
# pickle.dump({'data':data, 'labels':labels})
# f.close()


import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()