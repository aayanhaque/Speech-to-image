# AI Voice-to-Image Matcher

A Streamlit application that converts spoken words into matching images using speech recognition and image search capabilities.

## 🌟 Features

- **Voice Recording**: Record audio directly through the browser
- **Audio Upload**: Support for WAV and MP3 file uploads
- **Speech Recognition**: Converts speech to text using OpenAI's Whisper model
- **Intelligent Image Matching**: Finds relevant images based on spoken content
- **Real-time Processing**: Instant transcription and image retrieval
- **Responsive UI**: Beautiful and user-friendly interface

## 🛠️ Technologies Used

### Core Libraries
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework for AI models
- **Transformers**: Hugging Face's library for NLP tasks
- **NLTK**: Natural Language Processing toolkit

### Audio Processing
- **sounddevice**: Audio recording functionality
- **wavio**: WAV file handling
- **librosa**: Audio file processing
- **numpy**: Numerical computations

### Image Processing
- **Pixabay API**: Image search and retrieval
- **requests**: HTTP requests handling

## 📦 Dependencies

```txt
streamlit>=1.28.0
torch>=2.0.0
sounddevice>=0.4.6
wavio>=0.0.7
numpy>=1.24.0
transformers>=4.30.0
nltk>=3.8.1
librosa>=0.11.0
requests>=2.32.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up your Pixabay API key:
- Get your API key from [Pixabay](https://pixabay.com/api/docs/)
- Replace `PIXABAY_API_KEY` in `app.py` with your key

## 🎯 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Choose one of two options:
   - Click "Record Audio" to record 5 seconds of speech
   - Upload an audio file (WAV or MP3)

4. Click "Process Audio" to:
   - Transcribe the speech
   - Extract keywords
   - Find matching images

## 🔧 Technical Details

### Speech Recognition
- Uses OpenAI's Whisper model (tiny version)
- Supports multiple languages
- Optimized for real-time transcription

### Keyword Extraction
- Filters common words and stop words
- Identifies main subjects from speech
- Handles multiple word contexts

### Image Matching
- Uses Pixabay's extensive image database
- Supports high-resolution images
- Safe search enabled by default
- Returns multiple matching images

### UI Features
- Responsive grid layout
- Image hover effects
- Progress indicators
- Status messages
- Error handling

## 🔒 Security

- API keys are required for Pixabay integration
- Safe search enabled for image results
- Temporary file handling for uploads
- Error handling for all API calls

## ⚠️ Known Limitations

- Audio recording limited to 5 seconds
- Requires internet connection
- API rate limits apply
- Browser microphone access required for recording

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Pixabay for the image API
- Streamlit for the web framework
- The open-source community for various libraries

## How to run this project in complier/editor

streamlit run app.py