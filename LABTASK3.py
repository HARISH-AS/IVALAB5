import cv2
import numpy as np
import os
import csv

# Output folder for gender-detected images
output_folder = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\TASK3.3'
os.makedirs(output_folder, exist_ok=True)

# Load Haar Cascade for face detection and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to calculate enhanced geometric features
def geometric_feature_extraction(face_rect):
    x, y, w, h = face_rect
    face_width = w
    face_height = h
    jaw_width = w * 0.75  # Approximation for jaw width (usually 75% of face width)
    
    # Ratio of face width to height
    face_ratio = face_width / face_height
    
    # Additional features (example: jaw width, overall face shape)
    return face_width, face_height, jaw_width, face_ratio

# Function to detect hair features (long/short/bald)
def detect_hair(image, face_rect):
    x, y, w, h = face_rect
    # Crop the region around the head above the face to detect hair
    head_region = image[max(0, y - int(h * 0.5)):y, x:x + w]
    
    # Convert to grayscale and check for texture in the hair region
    gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    
    # Use a Gaussian blur to smooth the image and reduce noise
    blurred_head = cv2.GaussianBlur(gray_head, (5, 5), 0)
    
    # Threshold the image to find darker areas (possible hair)
    _, hair_mask = cv2.threshold(blurred_head, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate the percentage of hair pixels in the head region
    hair_pixels = np.sum(hair_mask == 255)
    total_pixels = hair_mask.size
    
    hair_density = hair_pixels / total_pixels
    
    # Refined threshold for detecting long hair
    if hair_density > 0.25:
        return "Long Hair"
    else:
        return "Short Hair or Bald"

# Function to check for facial hair (beard, mustache)
def detect_facial_hair(image, face_rect):
    x, y, w, h = face_rect
    mouth_region = image[y + int(0.6 * h):y + h, x:x + w]
    
    # Convert to grayscale and check for texture (facial hair detection)
    gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to smooth out the region
    blurred_mouth = cv2.GaussianBlur(gray_mouth, (5, 5), 0)
    
    # Threshold the region to identify darker textures (facial hair)
    _, mouth_mask = cv2.threshold(blurred_mouth, 75, 255, cv2.THRESH_BINARY_INV)
    
    # Count pixels indicating facial hair (texture around the mouth)
    facial_hair_pixels = np.sum(mouth_mask == 255)
    
    # Refined threshold for detecting facial hair
    if facial_hair_pixels > 800:  # Lower the threshold slightly
        return "Facial Hair"
    return "No Facial Hair"

# Function to classify gender using geometric and image features
def classify_gender(face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair):
    # Combine geometric and feature-based classification
    if hair_feature == "Long Hair" and facial_hair == "No Facial Hair":
        return "Female"
    elif hair_feature == "Short Hair or Bald" and facial_hair == "Facial Hair":
        return "Male"
    else:
        # Fallback to geometric features if ambiguous
        if face_ratio > 0.85 and jaw_width > 60:
            return "Male"
        else:
            return "Female"

# List to store gender detection results for each image
results = []

# Directory containing the images for gender detection
image_dir = 'D:\\FALL SEMESTER 2024-2025\\IVA LAB 5\\task3\\img_align_celeba\\img_align_celeba'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Process only the first 1000 images
image_files = image_files[:1000]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # Convert image to grayscale for face detection and feature extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Geometric feature extraction
        face_width, face_height, jaw_width, face_ratio = geometric_feature_extraction((x, y, w, h))

        # Detect hair features (long/short/bald)
        hair_feature = detect_hair(image, (x, y, w, h))
        
        # Detect facial hair (beard/mustache)
        facial_hair = detect_facial_hair(image, (x, y, w, h))

        # Classify gender based on all available features
        detected_gender = classify_gender(face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair)

        # Annotate the image with the detected gender
        cv2.putText(image, detected_gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the result image in the output folder
        output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_gender_detected.jpg")
        cv2.imwrite(output_path, image)

        # Save the results to list
        results.append([image_file, detected_gender, face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair])

# Save the detection results to a CSV file (optional)
csv_path = os.path.join(output_folder, "gender_identification_results.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name', 'Detected Gender', 'Face Width', 'Face Height', 'Jaw Width', 'Face Ratio', 'Hair Feature', 'Facial Hair'])
    writer.writerows(results)

print("Enhanced Gender Identification Complete! Results saved in:", output_folder)
