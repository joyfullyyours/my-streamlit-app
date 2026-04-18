import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace

# 1. Page Configuration (Must be FIRST)
st.set_page_config(page_title="MoodMirror AI", layout="centered")

# 2. Simplified Model Check
@st.cache_resource
def check_library():
    # We don't build manually anymore; we just ensure DeepFace is ready
    return True

library_ready = check_library()

# 3. UI Elements
st.title("😊 MoodMirror AI")
st.subheader("Real-Time Emotion Detection & Suggestions")

activities = {
    "happy": "Enjoy music, dance, or socialize!",
    "sad": "Take a walk or call a friend.",
    "angry": "Try deep breathing and relax.",
    "neutral": "Good time to study or work.",
    "surprise": "Try something new today!",
    "fear": "Relax and breathe slowly.",
    "disgust": "Take a short break."
}

# 4. Input Method
option = st.radio("Choose Input Method:", ["Upload Photo", "Use Camera"])

image = None
if option == "Upload Photo":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
else:
    captured = st.camera_input("Take a photo")
    if captured:
        image = Image.open(captured)

# 5. The "Smart" Analysis Logic
if image:
    st.image(image, caption="Input Image", use_container_width=True)
    img_array = np.array(image)

    with st.spinner("Analyzing emotion..."):
        try:
            # We let DeepFace handle the model building internally here
            result = DeepFace.analyze(
                img_path = img_array, 
                actions = ['emotion'],
                enforce_detection = False,
                detector_backend = 'opencv'
            )

            emotion = result[0]["dominant_emotion"]
            st.success(f"Detected Emotion: {emotion.capitalize()}")
            st.info(f"Suggestion: {activities.get(emotion, 'Stay positive!')}")

        except Exception as e:
            st.error("Could not analyze image. Try another photo with a clearer view of your face.")
