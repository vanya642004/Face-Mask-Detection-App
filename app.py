import streamlit as st
import gdown
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# === Download weights only if not already present ===
MODEL_WEIGHTS_ID = "1YnIqoBHN1t_pI1Ln9bbksZEg6w7GV_sL"  # <-- replace with real ID

if not os.path.exists("model_weights.h5"):
    gdown.download("https://drive.google.com/uc?id=1YnIqoBHN1t_pI1Ln9bbksZEg6w7GV_sL", "model_weights.h5", quiet=False)

# === Load model.json (already in GitHub) ===
with open("model.json", "r") as json_file:
    model = model_from_json(json_file.read())

# === Load weights ===
model.load_weights("model_weights.h5")

st.title("😷 Face Mask Detection App")
st.write("Upload an image to predict if the person is wearing a mask.")

# === Image upload ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "😷 MASK" if prediction < 0.5 else "❌ NO MASK"

    st.subheader(f"Prediction: {label}")
    st.progress(int((1 - prediction) * 100) if label == "😷 MASK" else int(prediction * 100))
