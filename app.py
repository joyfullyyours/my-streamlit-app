import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace

# 1. Page Configuration MUST be the first Streamlit command
st.set_page_config(page_title="MoodMirror AI", layout="centered")

# 2. Cached model loading
@st.cache_resource
def load_model():
    # Pre-loading the emotion model weights to speed up processing
    # We use "Emotion" with a capital E as required by the library
    return DeepFace.build_model("Emotion")

emotion_model = load_model()

# 3. Main UI Header
st.title("😊 MoodMirror AI")
st.subheader("Real-Time Emotion Detection & Suggestions")
st.write("Upload your photo or take one using the camera to see how you're feeling!")

# 4. Suggestions Dictionary
activities = {
    "happy": "Enjoy music, dance, or socialize!",
    "sad": "Take a walk or call a friend.",
    "angry": "Try deep breathing and relax.",
    "neutral": "Good time to study or work.",
    "surprise": "Try something new today!",
    "fear": "Relax and breathe slowly.",
    "disgust": "Take a short break."
}

# 5. Input Method Selection
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

# 6. Analysis Logic
if image:
    st.image(image, caption="Input Image", use_container_width=True)
    
    # Convert PIL Image to NumPy array (RGB) for DeepFace
    img_array = np.array(image)

    with st.spinner("Analyzing emotion..."):
        try:
            # Analyze using the pre-loaded model
            result = DeepFace.analyze(
                img_array,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            # Get the dominant emotion from the results
            emotion = result[0]["dominant_emotion"]

            # Display the results to the user
            st.success(f"Detected Emotion: {emotion.capitalize()}")
            st.info(f"Suggestion: {activities.get(emotion, 'Stay positive!')}")

        except Exception as e:
            st.error("Could not analyze image. Please ensure your face is clear and try again.")
