import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Intracranial Hemorrhage Detection", layout="centered")

# --- Helper Functions ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained Keras model."""
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(uploaded_file, target_size=(64, 64)):
    """Preprocesses the uploaded image for model prediction."""
    img = Image.open(uploaded_file)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1] as per ImageDataGenerator
    return img_array

# --- Main Application Logic ---
def main():
    st.title("🧠 Intracranial Hemorrhage Detection")
    st.write("Upload a brain CT scan image to detect potential intracranial hemorrhage.")

    # Define the model path (adjust if you saved it elsewhere in Drive)
    model_path_in_colab = "C:\\report AIDA\\brain_hemorrhage_cnn_model.h5"

    # Since Streamlit runs independently, we need to ensure the model path is accessible.
    # For local deployment, you would place 'brain_hemorrhage_cnn_model.h5' in the same directory as 'app.py'
    # For this example, we assume app.py is run in an environment where /content/drive/MyDrive is mounted or the model is copied.
    # In a real-world deployment (e.g., on a cloud platform), you would upload the model alongside your app.py.
    # For this Colab simulation, we'll use the path where we just saved it.
    # In a local run, you'd move the .h5 file to the same folder as app.py or adjust the path.

    # Check if the model exists (this check is more for local execution sanity)
    if not os.path.exists(model_path_in_colab):
        st.error(f"Model file not found at {model_path_in_colab}. Please ensure the model is saved correctly and accessible.")
        st.stop()

    # Load the model
    model = load_model(model_path_in_colab)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded CT Scan', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        processed_image = preprocess_image(uploaded_file)
        prediction = model.predict(processed_image)

        # Assuming binary classification with sigmoid activation
        if prediction[0][0] > 0.5:
            st.error(f"Prediction: **Hemorrhage Detected** (Confidence: {prediction[0][0]:.2f})")
        else:
            st.success(f"Prediction: **Normal Scan** (Confidence: {1 - prediction[0][0]:.2f})")

        st.write("---")
        st.write("**Note:** This is a demonstration for educational purposes. Always consult a medical professional for diagnosis.")


if __name__ == '__main__':
    main()
