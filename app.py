import streamlit as st
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# --- Main UI ---
st.title("🌿 AI Plant Disease Detection")
st.write(
    """
    Upload an image of a plant leaf, and the AI will analyze it. 
    (Model training on Kaggle in progress!)
    """
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    st.write("") # Spacer
    
    # --- Analyze Button (Non-functional) ---
    if st.button("🔍 Analyze Leaf"):
        with st.spinner("Analyzing... (Model integration pending)"):
            # This is where your model.predict() will go later
            import time
            time.sleep(2) # Simulate analysis time
            
        st.info("ℹ️ **Model Not Yet Connected!** This UI is ready. The next step is to train the model on Kaggle and link it.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Convolutional Neural Network (CNN) to detect plant diseases. "
    "The model is being trained on Kaggle using the PlantVillage dataset."
)