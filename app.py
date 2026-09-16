import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------
# Streamlit Configuration (MUST BE AT THE TOP)
# ---------------------------------------------------
st.set_page_config(
    page_title="Intracranial Hemorrhage Detection",
    page_icon="🧠",
    layout="centered"
)


# ---------------------------------------------------
# Load Model (Failsafe Keras 2/3 Loader)
# ---------------------------------------------------
@st.cache_resource
def load_model_file(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception:
            import tf_keras
            return tf_keras.models.load_model(model_path, compile=False)


# ---------------------------------------------------
# Preprocess Image
# ---------------------------------------------------
def preprocess_image(uploaded_file, target_size=(64, 64)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# ---------------------------------------------------
# Information Views
# ---------------------------------------------------
def show_hemorrhage_information():
    st.error("🚨 Possible Intracranial Hemorrhage Detected")
    st.subheader("⚠️ Immediate Precautions")
    st.markdown("""
    - **Seek emergency medical attention immediately.**
    - Do not ignore severe or sudden symptoms.
    - Keep the person resting and avoid unnecessary movement.
    - Do not give food, water, or medicines by mouth if the person is unconscious or vomiting.
    """)

    st.subheader("🚨 Warning Symptoms")
    st.markdown("""
    - Sudden severe headache
    - Vomiting or nausea
    - Loss of consciousness
    - Confusion or slurred speech
    - Weakness or numbness on one side of the body
    """)


def show_normal_information():
    st.success("✅ Model Prediction: Normal Scan")
    st.subheader("ℹ️ Important Information")
    st.markdown("""
    The model did not detect hemorrhage in this image.
    If symptoms persist, seek professional medical evaluation.
    """)


# ---------------------------------------------------
# Main Application
# ---------------------------------------------------
def main():
    st.title("🧠 Intracranial Hemorrhage Detection")
    st.write("Upload a brain CT scan image to obtain an AI-based educational prediction.")
    st.warning("It is not a medical diagnostic tool.")

    model_path = "brain_hemorrhage_cnn_model.h5"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = load_model_file(model_path)

    uploaded_file = st.file_uploader("Choose a brain CT scan image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.subheader("Uploaded CT Scan")
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("---")

        with st.spinner("Analyzing CT scan..."):
            processed_image = preprocess_image(uploaded_file)
            prediction = model.predict(processed_image, verbose=0)

        prob = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])

        st.subheader("🔍 Prediction Result")
        if prob > 0.5:
            st.error(f"🚨 Hemorrhage Detected\n\nModel Confidence: {prob:.2%}")
            show_hemorrhage_information()
        else:
            st.success(f"✅ Normal Scan\n\nModel Confidence: {(1.0 - prob):.2%}")
            show_normal_information()

if __name__ == "__main__":
    main()
