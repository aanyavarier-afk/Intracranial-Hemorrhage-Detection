import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ---------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Intracranial Hemorrhage Detection",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )
    return model


# ---------------------------------------------------
# Preprocess Image
# ---------------------------------------------------
def preprocess_image(uploaded_file, target_size=(64, 64)):

    img = Image.open(uploaded_file)

    # Convert image to RGB
    img = img.convert("RGB")

    # Resize
    img = img.resize(target_size)

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    return img_array


# ---------------------------------------------------
# Hemorrhage Precautions
# ---------------------------------------------------
def show_hemorrhage_information():

    st.error("🚨 Possible Intracranial Hemorrhage Detected")

    st.subheader("⚠️ Immediate Precautions")

    st.markdown("""
    - **Seek emergency medical attention immediately.**
    - Do not ignore severe or sudden symptoms.
    - Keep the person resting and avoid unnecessary movement.
    - Do not give food, water, or medicines by mouth if the person is
      unconscious, very drowsy, vomiting, or having difficulty swallowing.
    - Do not drive yourself if severe symptoms are present.
    - If the person becomes unconscious and is not breathing normally,
      contact emergency services and follow emergency first-aid instructions.
    """)

    st.subheader("🚨 Warning Symptoms")

    st.markdown("""
    Watch for symptoms such as:

    - Sudden severe headache
    - Vomiting
    - Loss of consciousness
    - Confusion
    - Difficulty speaking
    - Weakness or numbness on one side of the body
    - Difficulty walking
    - Vision problems
    - Seizures
    - Increasing sleepiness
    """)

    st.subheader("💊 Medication / Treatment Information")

    st.warning("""
    **Do not self-medicate based on this prediction.**

    Treatment for intracranial hemorrhage depends on the type, location,
    size, cause of the bleeding, and the patient's clinical condition.
    Doctors may use different treatments, including hospital monitoring,
    management of blood pressure, reversal of certain blood-thinning
    medicines when appropriate, procedures, or surgery.

    Medicines such as aspirin or anticoagulants should NOT be started
    or stopped without medical advice.
    """)

    st.subheader("🏥 What to Do")

    st.info("""
    Take the patient to an emergency department or contact your local
    emergency medical service as soon as possible.

    This application is only an educational demonstration and cannot
    confirm or rule out intracranial hemorrhage.
    """)


# ---------------------------------------------------
# Normal Scan Information
# ---------------------------------------------------
def show_normal_information():

    st.success("✅ Model Prediction: Normal Scan")

    st.subheader("ℹ️ Important Information")

    st.markdown("""
    The model did not detect hemorrhage in this image.

    However, a normal prediction does **not** guarantee that the brain
    is normal. If the patient has concerning symptoms, medical evaluation
    is still necessary.
    """)

    st.subheader("⚠️ When to Seek Medical Attention")

    st.markdown("""
    Seek urgent medical attention if there is:

    - Sudden severe headache
    - Loss of consciousness
    - Seizure
    - Sudden weakness or numbness
    - Difficulty speaking
    - Confusion
    - Repeated vomiting
    - Sudden vision problems
    """)


# ---------------------------------------------------
# Main Application
# ---------------------------------------------------
def main():

    st.title("🧠 Intracranial Hemorrhage Detection")

    st.write(
        "Upload a brain CT scan image to obtain an AI-based "
        "educational prediction."
    )

    st.warning(
        "It is not a medical diagnostic tool."
    )

    # ------------------------------------------------
    # Model Path
    # ------------------------------------------------

    model_path = "brain_hemorrhage_cnn_model.h5"

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found: {model_path}\n\n"
            "Place the trained model in the same folder as app.py."
        )

        st.stop()

    # ------------------------------------------------
    # Load Model
    # ------------------------------------------------

model_path = "brain_hemorrhage_cnn_model.h5"

    # ------------------------------------------------
    # Upload Image
    # ------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a brain CT scan image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Display uploaded image
        st.subheader("📷 Uploaded CT Scan")

        st.image(
            uploaded_file,
            caption="Uploaded CT Scan",
            use_container_width=True
        )

        st.write("---")

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        with st.spinner("Analyzing CT scan..."):

            processed_image = preprocess_image(uploaded_file)

            prediction = model.predict(
                processed_image,
                verbose=0
            )

        # Binary sigmoid output
        probability = float(prediction[0][0])

        st.subheader("🔍 Prediction Result")

        # ------------------------------------------------
        # Hemorrhage
        # ------------------------------------------------

        if probability > 0.5:

            confidence = probability

            st.error(
                f"🚨 Hemorrhage Detected\n\n"
                f"Model Confidence: {confidence:.2%}"
            )

            show_hemorrhage_information()

        # ------------------------------------------------
        # Normal
        # ------------------------------------------------

        else:

            confidence = 1 - probability

            st.success(
                f"✅ Normal Scan\n\n"
                f"Model Confidence: {confidence:.2%}"
            )

            show_normal_information()

        # ------------------------------------------------
        # Disclaimer
        # ------------------------------------------------

        st.write("---")

        st.subheader("⚕️ Medical Disclaimer")

        st.caption("""
        This AI model is intended for educational and research
        demonstration purposes only. The prediction should not be
        used to diagnose, treat, or rule out intracranial hemorrhage.
        CT scans should be interpreted by qualified healthcare
        professionals together with the patient's symptoms and
        clinical history.
        """)


# ---------------------------------------------------
# Run Application
# ---------------------------------------------------

if __name__ == "__main__":
    main()
