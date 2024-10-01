import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('C:\projects\Facial_Emotion_Recognition\my_model.h5')  # Update the path to your model

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):
    img = cv2.resize(image, (48, 48))  # Resize to model input shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize to [0, 1]
    img = np.reshape(img, (1, 48, 48, 1))  # Reshape for model
    return img

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces

st.title("Facial Emotion Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Detecting faces and classifying emotions...")

    # Detect faces
    faces = detect_faces(image_np)

    # Process each face and predict emotion
    predictions = []
    for (x, y, w, h) in faces:
        face_image = image_np[y:y+h, x:x+w]  # Extract face region
        processed_image = preprocess_image(face_image)
        prediction = model.predict(processed_image)
        predicted_class = emotion_labels[np.argmax(prediction)]
        predictions.append((predicted_class, (x, y, w, h)))  # Store emotion and coordinates

    # Display results
    for emotion, (x, y, w, h) in predictions:
        st.write(f'Predicted Emotion: {emotion} at location {x}, {y}, Width: {w}, Height: {h}')
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face

    # Show image with rectangles drawn around detected faces
    st.image(image_np, caption='Detected Faces with Emotions', use_column_width=True)
