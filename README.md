
# Rock Paper Scissors Game  

>![image](https://github.com/JanhaviP06/Rock-paper-scissor-gesture-game/blob/main/images/Project%20Banner.jpg)

## 🧠 Project Overview

This Game brings a modern twist to the classic _Rock Paper Scissors_ game using computer vision. Powered by OpenCV and MediaPipe, this interactive web app allows players to make hand gestures in front of a webcam to play against an AI opponent that detects gestures in real time.

---

## 🚀 Features

- 🎮 Play Rock, Paper, Scissors with your hand gestures
- 🧠 Gesture detection using MediaPipe and OpenCV
- 💻 Flask web application interface
- 📷 Real-time webcam integration
- 🧪 Easy-to-use and responsive web interface

---


## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Python, Flask  
- **Computer Vision**: OpenCV, MediaPipe  

---

### 📌 Project Workflow

#### Step 1: Prepare Image Data
Organize your gesture images like this:

```bash
image_data/
├── Rock/
│   ├── img1.jpg
│   ├── img2.jpg
├── Paper/
│   ├── img1.jpg
├── Scissors/
│   ├── img1.jpg
```
Each subfolder name (Rock, Paper, Scissors) becomes the label for the images inside it.

#### Step 2: Extract Landmarks and Create Dataset
Run the script to process all images and extract hand landmarks using MediaPipe.
```bash
python createData.py
```
This will generate a CSV file named hand_landmarks_dataset.csv.

#### Step 3: Train the Model
Train a neural network using the extracted landmark dataset:

```bash
python train_model.py
```
This will:

- Encode and normalize the data

- Train a model for classification

- Save the trained model as hand_landmark_model.h5

- Save the scaler used in preprocessing as scaler.pkl

### Model Architecture
- Input: 42 values (21 x, y coordinates from landmarks)
- Hidden Layers:
  - Dense(128, ReLU) + Dropout(0.5)

  - Dense(64, ReLU)

- Output: 3 neurons (Softmax) for multi-class classification (Rock, Paper, Scissors)



## 📁 Files in the Project

- `hand_landmark_data.py`: Script to extract hand landmarks and save to CSV.
- `train_model.py`: Script to train and save the gesture classification model.
- `hand_landmarks_dataset.csv`: Generated dataset of features and labels.
- `hand_landmark_model.h5`: Trained deep learning model.
- `scaler.pkl`: Scaler used for feature normalization.

---

## 🎯 How It Works
- The app captures real-time webcam frames.
- MediaPipe detects hand landmarks.
- Based on finger positions, the gesture is classified as Rock, Paper, or Scissors.
- AI makes its move, and the winner is decided.

---

## ⚙️ Installation and Setup

1. **Clone the repository**

```bash
git clone [https://github.com/JanhaviP06/Rock-paper-scissor-gesture-game.git]
cd Rock-paper-scissor-gesture-game
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
```
3.**Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the app**
```bash
python run.py
```
Open your browser
```bash
Visit: http://127.0.0.1:5000/
```
---

## ✨ Future Improvements
- Add sound effects and animations
- Scoreboard system
- Multiplayer over local network
- Mobile optimization

---

## 🤝 Connect with Me

Feel free to reach out for collaboration, feedback, or opportunities!
- 📧 Gmail: [janhavi.phulavare06@gmail.com](mailto:janhavi.phulavare06@gmail.com)  
- 💼 LinkedIn: [linkedin.com/in/janhavi-phulavare](https://www.linkedin.com/in/janhavi-phulavare)

⭐ Star the repo if you found it helpful!

