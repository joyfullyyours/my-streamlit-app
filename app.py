import numpy as np
from PIL import Image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Convert the uploaded file to a PIL Image
    img = Image.open(uploaded_file)
    
    # 2. Convert PIL Image to a NumPy array (RGB)
    img_array = np.array(img)

    try:
        # 3. Pass the array directly to DeepFace
        # We set enforce_detection=False so it doesn't crash if a face isn't perfectly clear
        results = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
        
        # Display results
        st.write(f"Detected Emotion: {results[0]['dominant_emotion']}")
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

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
