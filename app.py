import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

# --- Load model architecture ---
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# --- Load weights ---
model.load_weights("model_weights.h5")

st.title("😷 Face Mask Detection App")

# Toggle between Webcam and Image Upload
option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img = image.resize((150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"

        st.subheader(f"Prediction: {label}")
        st.progress(int((1 - prediction) * 100) if label == "😷 MASK" else int(prediction * 100))

else:
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        success, frame = camera.read()
        if not success:
            st.error("Camera not detected.")
            break

        img = cv2.resize(frame, (150, 150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"
        color = (0, 255, 0) if label == "😷 MASK" else (255, 0, 0)

        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

