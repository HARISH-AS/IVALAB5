import cv2
import numpy as np

# Load the video from the specified path
input_path = r'D:\FALL SEMESTER 2024-2025\IVA LAB 5\3029469-hd_1920_1080_24fps.mp4'
cap = cv2.VideoCapture(input_path)

# Get video frame width, height, and FPS for saving the output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save the output video in the specified location
output_path = r'D:\FALL SEMESTER 2024-2025\IVA LAB 5\output_motion_detection.avi'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

# Frame differencing
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
event_frames = []  # List to store event timestamps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for comparison
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    diff_frame = cv2.absdiff(prev_frame_gray, current_frame_gray)

    # Apply threshold to isolate regions with significant motion
    _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Find contours of moving areas
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Event detection based on contour size (significant motion)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust threshold for "significant" motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Annotate the frame
            cv2.putText(frame, "Event Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            event_frames.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Store event time in seconds

    # Write the frame with highlighted motion to the output video
    out.write(frame)

    # Update the previous frame
    prev_frame_gray = current_frame_gray.copy()

# Release video objects
cap.release()
out.release()

# Output event timestamps
print(f"Event timestamps (in seconds): {event_frames}")




