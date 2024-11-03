import cv2
import numpy as np
import os
import pandas as pd
import csv

# Paths
dataset_path = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\task3\\img_align_celeba\\img_align_celeba'
csv_file_path = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\task3\\list_attr_celeba.csv'
output_folder = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\TASK3'

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to calculate geometric features
def geometric_feature_extraction(face_rect):
    x, y, w, h = face_rect
    face_width = w
    face_height = h
    jaw_width = w * 0.8  # Placeholder for jaw width (adjust as needed)
    return face_width, face_height, jaw_width

# Function to extract LBP (Local Binary Patterns)
def extract_lbp(gray_image):
    lbp = cv2.Laplacian(gray_image, cv2.CV_64F)  # Basic texture extraction using Laplacian for edges
    return lbp

# Improved gender classification function using multiple features
def classify_gender(row, face_width, face_height, jaw_width):
    # Use features from the dataset to enhance classification
    male_features = ['Goatee', 'Mustache', 'No_Beard', 'Bushy_Eyebrows', '5_o_Clock_Shadow']
    female_features = ['Heavy_Makeup', 'Wearing_Lipstick', 'Arched_Eyebrows']

    male_score = sum([1 for feature in male_features if row[feature] == 1])
    female_score = sum([1 for feature in female_features if row[feature] == 1])

    # Use hair-related features as well
    if row['Bald'] == 1:
        male_score += 2  # Baldness is more common in males

    if row['Blond_Hair'] == 1 or row['Brown_Hair'] == 1:
        female_score += 1  # Females often have specific hair colors

    # Use geometric face ratio
    face_ratio = face_width / face_height
    if face_ratio > 0.85 and jaw_width > 60:  # Adjusted thresholds
        male_score += 1
    else:
        female_score += 1

    # Final decision based on combined scores
    if male_score > female_score:
        return "Male"
    else:
        return "Female"

# Create output CSV for results
with open(f'{output_folder}\\gender_identification_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name', 'Detected Gender', 'Face Width', 'Face Height', 'Jaw Width'])

    # Load the CSV with gender labels
    attr_df = pd.read_csv(csv_file_path)
    subset_df = attr_df.head(5000)  # Only first 5000 images

    # Loop through images in the subset
    for index, row in subset_df.iterrows():
        image_name = row['image_id']
        image_path = os.path.join(dataset_path, image_name)

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for face_rect in faces:
                # Geometric feature extraction
                face_width, face_height, jaw_width = geometric_feature_extraction(face_rect)

                # Texture feature extraction (LBP)
                lbp_features = extract_lbp(gray_image)

                # Gender classification using multiple features
                predicted_gender = classify_gender(row, face_width, face_height, jaw_width)

                # Draw result on the image
                x, y, w, h = face_rect
                cv2.putText(image, predicted_gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Save the processed image
                cv2.imwrite(f'{output_folder}\\{image_name}_gender_detected.jpg', image)

                # Write results to CSV
                writer.writerow([image_name, predicted_gender, face_width, face_height, jaw_width])

# Output the final results
print("Enhanced Gender Identification Task Complete!")

