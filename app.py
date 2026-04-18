import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def load_model():
    # This "pre-loads" the emotion model into memory properly
    return DeepFace.build_model("emotion")

# Call the function to store the model in memory
emotion_model = load_model()
# -------------------

st.set_page_config(page_title="MoodMirror AI", layout="centered")

st.title("😊 MoodMirror AI")
st.subheader("Real-Time Emotion Detection & Suggestions")

st.write("Upload your photo or take one using camera.")

activities = {
    "happy": "Enjoy music, dance, or socialize!",
    "sad": "Take a walk or call a friend.",
    "angry": "Try deep breathing and relax.",
    "neutral": "Good time to study or work.",
    "surprise": "Try something new today!",
    "fear": "Relax and breathe slowly.",
    "disgust": "Take a short break."
}

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

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    img_array = np.array(image)

    with st.spinner("Analyzing emotion..."):
        try:
            result = DeepFace.analyze(
                img_array,
                actions=['emotion'],
                enforce_detection=False,
                models={'emotion':emotion_model} #Use the cached model
            )

            emotion = result[0]["dominant_emotion"]

            st.success("Detected Emotion: " + emotion.capitalize())
            st.info("Suggestion: " + activities.get(emotion, "Stay positive!"))

        except:
            st.error("Could not analyze image. Try another photo.")
