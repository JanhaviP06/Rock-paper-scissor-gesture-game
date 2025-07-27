# import os
# import pickle
# import mediapipe as mp
# import cv2
# import pandas as pd
# import numpy as np
#
# # Initialize Mediapipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# # Dataset directories
# DATA_DIR = './image_data'
# SAVE_DIR = './processed2'  # ✅ Folder to save processed images
#
# # Ensure SAVE_DIR exists
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# data = []
# labels = []
#
# for dir_ in os.listdir(DATA_DIR):
#     class_dir = os.path.join(SAVE_DIR, dir_)
#     os.makedirs(class_dir, exist_ok=True)  # ✅ Create class folder if not exists
#
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#         x_ = []
#         y_ = []
#
#         img_path_full = os.path.join(DATA_DIR, dir_, img_path)
#         img = cv2.imread(img_path_full)
#
#         if img is None:
#             continue  # Skip if the image is not found
#
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#
#         # Get image size
#         h, w, _ = img.shape
#         white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # ✅ Create a white background
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # ✅ Draw landmarks on the white background
#                 mp_drawing.draw_landmarks(
#                     white_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
#
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)
#
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#
#             data.append(data_aux)
#             labels.append(dir_)
#
#             # ✅ Save processed image with landmarks on white background
#             save_path = os.path.join(class_dir, img_path)  # Save in respective class folder
#             cv2.imwrite(save_path, white_bg)
#
# hands.close()
#
# # Convert to Pandas DataFrame
# df = pd.DataFrame(data)
#
# # Add labels column
# df['label'] = labels
#
# # Save as CSV file
# df.to_csv('hand_landmarks_dataset.csv', index=False)
#
# print(f"✅ Dataset saved as 'hand_landmarks_dataset.csv'.")
# print(f"✅ Processed images saved in '{SAVE_DIR}/'.")


import os
import cv2
import mediapipe as mp
import csv

# Paths
IMAGE_FOLDER = "image_data"
OUTPUT_CSV = "hand_landmarks_dataset.csv"

# Mediapipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Prepare CSV
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    # CSV Header: 42 coordinates (x, y for 21 landmarks) + label
    header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    writer.writerow(header)

    # Process Each Image
    for label in os.listdir(IMAGE_FOLDER):  # Each subfolder is a label
        label_path = os.path.join(IMAGE_FOLDER, label)
        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # Read and Process Image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read {img_path}. Skipping...")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Extract Hand Landmarks
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Use the first detected hand
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])  # Use (x, y) coordinates only

                # Write to CSV
                writer.writerow([label] + landmarks)
                print(f"Processed {img_path}")
            else:
                print(f"No hand landmarks detected in {img_path}. Skipping...")

# Release Mediapipe
hands.close()
print(f"Landmark dataset saved to {OUTPUT_CSV}")
