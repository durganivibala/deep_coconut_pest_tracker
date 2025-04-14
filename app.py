import os
import gdown
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Streamlit UI
st.title("Deep Coconut Pest Tracker🌿")
st.write("Upload an image of a coconut leaf to check its infestation level.")

# Model path and download if needed
MODEL_PATH = "coconut_leaf_classifier.keras"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1UExt0qM7ATbrbOqLm2LexNIKOfFe0Wx7"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Class labels
class_labels = {0: "MEDIUM", 1: "HIGH", 2: "LOW"}

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    result = class_labels.get(predicted_class, "Unknown")

    # Show result
    st.success(f"The infestation level is **{result}**.")
