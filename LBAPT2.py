import cv2
import numpy as np
import csv

# Path to the input image and output folder
image_path = "C:\\Users\\ASUS\\Downloads\\varunfake.jpg"
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
cv2.imwrite(f'{output_folder}\\Task2-preprocessing6.jpg', skin)

# --- Step 2: Gesture Analysis - Detect facial features using OpenCV (Haar Cascade) ---
# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert image to grayscale for face detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adjust face detection parameters: lowering scaleFactor and minNeighbors for smaller faces
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

detected_emotions = []  # List to store emotions of each individual
emotion_labels = []  # List to store individual face coordinates and their emotions

# Loop through each detected face and draw a rectangle around the face
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    face_roi = gray_image[y:y+h, x:x+w]

    # Analyze the region of the mouth to decide emotion
    mouth_region = gray_image[y + int(h / 2):y + h, x:x + w]
    mouth_open = np.mean(mouth_region)  # Approximate threshold for detecting mouth openness

    # Refine thresholds for happy and neutral detection
    if mouth_open > 85:  # Lowered threshold for detecting subtle smiles (slightly open mouth)
        emotion = "Happy"  # Detecting smiling/open mouth
    elif 55 <= mouth_open <= 85:  # Adjusted range for neutral expressions (closed or subtle mouth movements)
        emotion = "Neutral"  # Assigning neutral for closed, non-frowning mouth
    else:
        emotion = "Sad"  # Detecting frowning or sad face

    # Draw the emotion label on the image
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Append the detected emotion to the list
    detected_emotions.append(emotion)
    emotion_labels.append([x, y, w, h, emotion])

    # Draw rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save the output with face detection and classified emotions
cv2.imwrite(f'{output_folder}\\Task2.2-gesture_analysis_optimized_v6.jpg', image)

# --- Step 3: Categorize Overall Sentiment ---
# Count the occurrences of each emotion
happy_count = detected_emotions.count("Happy")
sad_count = detected_emotions.count("Sad")
neutral_count = detected_emotions.count("Neutral")

# Determine the majority sentiment
if happy_count > sad_count and happy_count > neutral_count:
    overall_sentiment = "Majority Happy"
elif sad_count > happy_count and sad_count > neutral_count:
    overall_sentiment = "Majority Sad"
else:
    overall_sentiment = "Neutral Crowd"

# Save the overall sentiment to a CSV file with detailed emotion labels
with open(f'{output_folder}\\Task2-6-overall_sentiment_optimized_v3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Face X', 'Face Y', 'Width', 'Height', 'Emotion'])
    writer.writerows(emotion_labels)  # Writing each face's coordinates and detected emotion
    writer.writerow([])
    writer.writerow(['Happy Count', 'Sad Count', 'Neutral Count', 'Overall Sentiment'])
    writer.writerow([happy_count, sad_count, neutral_count, overall_sentiment])

# Output the final results
print("Image Processing and Sentiment Analysis Complete!")
print(f"Detected emotions: {detected_emotions}")
print(f"Overall Sentiment: {overall_sentiment}")
