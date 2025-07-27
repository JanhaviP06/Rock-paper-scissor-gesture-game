
# ✋ Hand Gesture Recognition using Mediapipe and Deep Learning

## 📝 Project Description

**Hand Gesture Recognition using Mediapipe and Deep Learning** is a computer vision project that aims to recognize human hand gestures based on landmark detection and classification using a neural network. This system focuses on gestures like **Rock, Paper, and Scissors**, making it a foundational step toward building gesture-controlled applications such as games, sign language interpreters, and human-computer interaction tools.

### 📌 Project Workflow

#### 1. Data Preparation (`hand_landmark_data.py`)
- Reads images from folders representing gestures (`rock/`, `paper/`, `scissors/`).
- Uses **MediaPipe Hands** to extract 21 hand landmarks per image.
- Flattens x and y coordinates of the landmarks and saves them into a structured CSV file.

#### 2. Model Training (`train_model.py`)
- Loads the dataset and encodes the labels.
- Normalizes the feature data using `StandardScaler`.
- Builds a Keras deep learning model to classify gestures.
- Saves the trained model (`hand_landmark_model.h5`) and scaler (`scaler.pkl`) for deployment.

## 🧠 Technologies Used
- Python
- MediaPipe by Google
- TensorFlow / Keras
- Scikit-learn (for scaling and label encoding)
- NumPy and Pandas

## 📁 Files in the Project
- `hand_landmark_data.py`: Script to extract hand landmarks and save to CSV.
- `train_model.py`: Script to train and save the gesture classification model.
- `hand_landmarks_dataset.csv`: Generated dataset of features and labels.
- `hand_landmark_model.h5`: Trained deep learning model.
- `scaler.pkl`: Scaler used for feature normalization.

## 📊 Output Example (Landmark Dataset)
| label | x0 | y0 | x1 | y1 | ... | x20 | y20 |
|-------|----|----|----|----|-----|-----|-----|
| rock  | .. | .. | .. | .. | ... | ..  | ..  |
| paper | .. | .. | .. | .. | ... | ..  | ..  |
| ...   | .. | .. | .. | .. | ... | ..  | ..  |

## 🚀 Future Scope
- Extend to more gestures (e.g., numbers, signs, emojis).
- Integrate with real-time webcam feed for live predictions.
- Build interactive games or apps using the model.

## 📌 License
This project is for educational purposes only.
