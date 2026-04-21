# Sign Language Recognition (Hand Tracking + ML)

## What is this?
This is a small project I built to recognize hand signs (like letters) using the webcam.

The idea is simple:
- detect the hand
- extract its shape (landmarks)
- train a model
- predict the sign in real time

---

## What I used
- Python  
- OpenCV → for camera and image processing  
- MediaPipe → to detect hand landmarks  
- Scikit-learn → Random Forest model  

---

## How it works

1. **Collect data**
   I capture images of different hand signs using the webcam.

2. **Feature extraction**
   MediaPipe gives me hand landmarks (basically coordinates of fingers and joints).

3. **Dataset**
   I store these coordinates with labels (A, B, etc.).

4. **Training**
   I train a Random Forest classifier on this data.

5. **Prediction**
   The webcam runs in real time and predicts the hand sign.

---

## Files

- `collect_img.py` → collect images from webcam  
- `create_dataset.py` → convert images to landmark data  
- `train_classifier.py` → train the model  
- `inference_classifier.py` → run real-time prediction  
- `data/` → saved images  
- `data.pickle` → processed dataset  
- `model.p` → trained model  

---

## Notes
- Accuracy depends a lot on the dataset (lighting, angles, etc.)
- Works well for simple signs, but not full words

