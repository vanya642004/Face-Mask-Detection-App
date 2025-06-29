import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

st.title("😷 Real-Time Mask Detection via Webcam")

model = load_model("model.h5")

# Start camera
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.write("Camera not detected.")
        break

    # Preprocess frame
    img = cv2.resize(frame, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"
    color = (0, 255, 0) if label == "😷 MASK" else (255, 0, 0)

    # Overlay result
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
