import streamlit as st
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import os
import gdown

st.title("😷 Real-Time Face Mask Detection")

# === Google Drive ID of TFLite model ===
TFLITE_MODEL_ID = "1YnIqoBHN1t_pI1Ln9bbksZEg6w7GV_sL"  # replace if needed

# === Download model.tflite if not present ===
if not os.path.exists("model.tflite"):
    gdown.download(f"https://drive.google.com/uc?id={TFLITE_MODEL_ID}", "model.tflite", quiet=False)

# === Load TFLite model ===
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Start webcam stream ===
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.warning("Could not access webcam.")
        break

    # Preprocess frame
    img = cv2.resize(frame, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"
    color = (0, 255, 0) if label == "😷 MASK" else (255, 0, 0)

    # Overlay result
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

camera.release()
