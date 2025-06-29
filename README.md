# 😷 Face Mask Detection App

This project is a real-time face mask detector built using a Convolutional Neural Network (CNN) and deployed with Streamlit.

## 🚀 Features

- Classifies whether a person in an image is **Wearing a Mask** or **Not Wearing a Mask**.
- Uses a trained CNN model (`model.h5`)
- Deployed using **Streamlit Cloud** or local webcam
- Real-time detection from webcam (local deployment)
- Scalable and customizable

## 📁 Files

- `app.py` - Streamlit app interface
- `model.h5` - Pre-trained CNN model (upload in same folder)
- `requirements.txt` - List of dependencies

## 🛠️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py
