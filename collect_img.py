import os
import cv2

# Directory for storing collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parameters
number_of_classes = 26
dataset_size = 100

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0, or test with 1, 2, etc., if needed
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

for class_id in range(number_of_classes):
    # Create a directory for each class
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}')
    
    # Display "Ready" message
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Wait for 'q' key to proceed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Start capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break
        
        # Save the frame
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
        
        # Show progress on screen
        cv2.putText(frame, f'Capturing image {counter}/{dataset_size}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Delay for visibility
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()
