import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("😷 Face Mask Detection with TFLite")
st.write("Upload an image and I’ll tell you if the person is wearing a mask.")

# === Load the .tflite model ===
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Upload Image ===
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    # Result
    label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"
    st.subheader(f"Prediction: {label}")
    st.progress(int((1 - prediction) * 100) if label == "😷 MASK" else int(prediction * 100))
