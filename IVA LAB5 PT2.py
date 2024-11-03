import cv2
import dlib
import numpy as np
import csv

# Path to the input image and output folder
image_path = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\TASK2.jpg'
output_folder = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5'

# Load the image
image = cv2.imread(image_path)

# --- Step 1: Preprocessing: Skin-color-based detection to detect faces and hands ---
# Convert the image to HSV for skin color detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Threshold the image to get only skin-colored regions
skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Perform morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

# Bitwise-AND mask and original image to extract the skin regions
skin = cv2.bitwise_and(image, image, mask=skin_mask)

# Save the preprocessed image with skin detection
cv2.imwrite(f'{output_folder}\Task2-preprocessing.jpg', skin)

# --- Step 2: Gesture Analysis - Detect facial features using dlib ---
# Load the facial landmark predictor from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this file is in the correct path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Convert image to grayscale for facial landmark detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray_image)

detected_emotions = []  # List to store emotions of each individual

# Loop through each detected face and find facial landmarks
for face in faces:
    landmarks = predictor(gray_image, face)

    # Draw the facial landmarks on the face
    for n in range(36, 48):  # Eyes region
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    for n in range(48, 68):  # Mouth region
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # --- Step 3: Emotion Classification ---
    # Calculate the curvature of the mouth (points 48-54 for the upper lip, 55-61 for lower)
    mouth_up = landmarks.part(51).y  # Upper lip point
    mouth_down = landmarks.part(57).y  # Lower lip point

    # Classify emotion based on mouth curvature
    if mouth_up < mouth_down:  # Smiling if upper lip is higher than lower lip
        emotion = "Happy"
    else:  # Frowning or neutral
        emotion = "Sad"
    
    # Draw the emotion label on the image
    cv2.putText(image, emotion, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Append the detected emotion to the list
    detected_emotions.append(emotion)

# Save the output with facial landmarks and classified emotions
cv2.imwrite(f'{output_folder}\Task2-gesture_analysis.jpg', image)

# --- Step 4: Categorize Overall Sentiment ---
# Count the occurrences of each emotion
happy_count = detected_emotions.count("Happy")
sad_count = detected_emotions.count("Sad")

# Determine the majority sentiment
if happy_count > sad_count:
    overall_sentiment = "Majority Happy"
elif sad_count > happy_count:
    overall_sentiment = "Majority Sad"
else:
    overall_sentiment = "Neutral Crowd"

# Save the overall sentiment to a CSV file
with open(f'{output_folder}\Task2-overall_sentiment.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Happy Count', 'Sad Count', 'Overall Sentiment'])
    writer.writerow([happy_count, sad_count, overall_sentiment])

# Output the final results
print("Image Processing and Sentiment Analysis Complete!")
print(f"Detected emotions: {detected_emotions}")
print(f"Overall Sentiment: {overall_sentiment}")


