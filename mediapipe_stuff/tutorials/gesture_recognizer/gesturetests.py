import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cap.read()  # Capture a single frame

# Save the captured image
cv2.imwrite('photo.jpg', frame)

cap.release()  