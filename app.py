import streamlit as st
import torch
import sounddevice as sd
import wavio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tempfile
import os
import requests
import random
import librosa

# Pixabay API configuration
PIXABAY_API_KEY = "50466533-359243a727b90051e9fd5f913"  # Pixabay API key
PIXABAY_API_URL = "https://pixabay.com/api/"

# Page configuration
st.set_page_config(
    page_title="AI Voice-to-Image Matcher",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 3em;
            margin-top: 1em;
        }
        .status-box {
            padding: 1em;
            border-radius: 0.5em;
            margin: 1em 0;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
        }
        .info-box {
            background-color: #cce5ff;
            color: #004085;
        }
        .warning-box {
            background-color: #fff3cd;
            color: #856404;
        }
        .image-container {
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .image-container img {
            width: 100%;
            height: auto;
            transition: transform 0.3s ease;
        }
        .image-container img:hover {
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'images' not in st.session_state:
    st.session_state.images = None

# Add predefined image collections
PRESET_IMAGES = {
    "positive": [
        "https://images.unsplash.com/photo-1490730141103-6cac27aaab94",  # Sunrise
        "https://images.unsplash.com/photo-1552083375-1447ce886485",  # Happy people
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131",  # Nature
        "https://images.unsplash.com/photo-1506126613408-eca07ce68773"   # Celebration
    ],
    "negative": [
        "https://images.unsplash.com/photo-1477346611705-65d1883cee1e",  # Moody landscape
        "https://images.unsplash.com/photo-1499346030926-9a72daac6c63",  # Rain
        "https://images.unsplash.com/photo-1516410529446-2c777cb7366d",  # Dark clouds
        "https://images.unsplash.com/photo-1492783391015-11d7cd0d6e3b"   # Dramatic scene
    ],
    "neutral": [
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",  # Calm beach
        "https://images.unsplash.com/photo-1476820865390-c52aeebb9891",  # Peaceful landscape
        "https://images.unsplash.com/photo-1502082553048-f009c37129b9",  # Nature
        "https://images.unsplash.com/photo-1505765050516-f72dcac9c60e"   # Serene scene
    ]
}

def get_matching_images(sentiment_score):
    """Get matching images based on sentiment score"""
    if sentiment_score >= 0.5:
        return PRESET_IMAGES["positive"]
    elif sentiment_score <= -0.5:
        return PRESET_IMAGES["negative"]
    else:
        return PRESET_IMAGES["neutral"]

# Replace the get_images_from_unsplash function with this simpler version
def get_images_from_sentiment(sentiment):
    try:
        compound_score = sentiment['compound']
        return get_matching_images(compound_score)
    except Exception as e:
        st.error(f"Error getting images: {str(e)}")
        return None

# Download NLTK data
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

def get_images_from_pixabay(query, per_page=4):
    """Get images from Pixabay API based on the query"""
    try:
        # If no API key is provided, use a default image
        if PIXABAY_API_KEY == "YOUR_PIXABAY_API_KEY":
            return [
                "https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_1280.jpg",
                "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg",
                "https://cdn.pixabay.com/photo/2019/08/19/07/45/dog-4415649_1280.jpg",
                "https://cdn.pixabay.com/photo/2016/03/27/20/51/dogs-1284238_1280.jpg"
            ]
        
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "per_page": per_page,
            "image_type": "photo",
            "safesearch": "true",
        }
        response = requests.get(PIXABAY_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["hits"]:
                return [img["largeImageURL"] for img in data["hits"]]
            else:
                st.warning(f"No images found for '{query}'. Showing default images.")
                return get_images_from_pixabay("general")
        else:
            st.error(f"Error fetching images: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error accessing Pixabay API: {str(e)}")
        return None

def extract_keywords(text):
    """Extract meaningful keywords from text"""
    try:
        # Simple word splitting and filtering
        words = text.lower().split()
        # Filter out short words and common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 2 and word not in common_words]
        
        # Return the first meaningful word or 'general' if none found
        return keywords[0] if keywords else 'general'
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        return 'general'

# Initialize models
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, whisper_model

def record_audio(duration=5, sample_rate=16000):
    try:
        st.info("🎤 Recording...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None, None

def save_audio(audio_data, sample_rate):
    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "recorded_audio.wav")
        wavio.write(temp_path, audio_data, sample_rate, sampwidth=2)
        return temp_path
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def transcribe_audio(audio_path, processor, model):
    try:
        # Load audio file
        audio_input, _ = librosa.load(audio_path, sr=16000)
        
        # Process audio input
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription
        predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def enhance_prompt(text, sentiment):
    # Enhance prompt based on sentiment
    compound_score = sentiment['compound']
    
    if compound_score >= 0.5:
        mood = "happy, cheerful, bright"
    elif compound_score <= -0.5:
        mood = "dark, moody, dramatic"
    else:
        mood = "neutral, calm"
    
    # Extract key nouns and adjectives for better image search
    words = text.split()
    search_terms = [word for word in words if len(word) > 3]  # Simple filtering for significant words
    if search_terms:
        enhanced_prompt = f"{' '.join(search_terms[:3])} {mood}"
    else:
        enhanced_prompt = mood
    
    return enhanced_prompt

def main():
    st.title("🎤 AI Voice-to-Image Matcher")
    st.markdown("Speak or upload audio to find matching images!")
    
    # Initialize models
    with st.spinner("🚀 Loading AI models..."):
        setup_nltk()
        processor, whisper_model = load_models()
    
    # Audio input section
    st.subheader("1️⃣ Record or Upload Audio")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Record Audio (5s)"):
            audio_data, sample_rate = record_audio()
            if audio_data is not None:
                audio_path = save_audio(audio_data, sample_rate)
                if audio_path:
                    st.session_state.audio_file = audio_path
                    st.success("✅ Recording completed!")
    
    with col2:
        uploaded_file = st.file_uploader("📁 Upload Audio File", type=['wav', 'mp3'])
        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file = temp_path
            st.success("✅ File uploaded successfully!")
    
    # Process audio if available
    if st.session_state.audio_file and st.button("🔄 Process Audio"):
        # Transcribe audio
        st.subheader("2️⃣ Transcription")
        transcription = transcribe_audio(st.session_state.audio_file, processor, whisper_model)
        
        if transcription:
            st.session_state.transcription = transcription
            st.info(f"🗣️ Transcription: {transcription}")
            
            # Extract keywords and get matching images
            st.subheader("3️⃣ Matching Images")
            keyword = extract_keywords(transcription)
            st.info(f"🔍 Searching for images of: {keyword}")
            
            images = get_images_from_pixabay(keyword)
            if images:
                st.session_state.images = images
                cols = st.columns(2)
                for idx, image_url in enumerate(images):
                    with cols[idx % 2]:
                        st.markdown(f'<div class="image-container"><img src="{image_url}" alt="Matching image {idx+1}"></div>', unsafe_allow_html=True)
                st.success("✨ Process completed successfully!")

if __name__ == "__main__":
    main() 