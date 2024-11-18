from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import streamlit as st
from PIL import Image

import gdown
import os


# Path to save model
MODEL_PATH = "coconut_leaf_classifier.keras"

# Check if the model file already exists locally, if not, download it
if not os.path.exists(MODEL_PATH):
    # Google Drive shareable link
    url = "https://drive.google.com/uc?export=download&id=1b0LqliSna1qEiPiGidlf80KIlICw39uB"

    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)


# Class labels
class_labels = {0: "Infected", 1: "Healthy"}
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
MODEL_PATH = "coconut_leaf_classifier.keras"
model = load_model(MODEL_PATH)

# Class labels
class_labels = {0: "Infected", 1: "Healthy"}

# Streamlit UI
st.title("Deep Coconut Pest Tracker🌿")
st.write("Upload an image of a coconut leaf to check if it's infected or healthy.")

# Image upload widget
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((128, 128))  # Resize to match the model input
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = 1 if prediction > 0.5 else 0  # Threshold: 0.5
    result = class_labels[predicted_class]

    # Display the result
    st.success(f"The leaf is **{result}**.")
