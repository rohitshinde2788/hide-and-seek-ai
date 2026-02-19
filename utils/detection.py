import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_faces(gray):
    return face_cascade.detectMultiScale(gray,1.3,5)

def detect_eyes(face_gray):
    return eye_cascade.detectMultiScale(face_gray)
