# 🎙️ Nirva Audio Lab

Nirva Audio Lab is a powerful audio processing application that provides comprehensive audio analysis and transcription capabilities. Built with Python and Streamlit, it offers an intuitive interface for processing audio files with advanced features.

## ✨ Features

- **✂️ Silence Detection and Trimming**
  - Automatically detect and remove silence segments
  - Visual timeline representation of audio content
  - Adjustable silence threshold and minimum length
  - Download trimmed audio files

- **🔄 Audio Segmentation**
  - Split audio into meaningful segments
  - Intelligent boundary detection
  - Preserve context between segments

- **🗣️ Advanced Transcription**
  - Speaker identification and diarization
  - High-accuracy speech-to-text conversion
  - Support for multiple audio formats (WAV, MP3, M4A)

- **😊 Emotion Detection**
  - Analyze speaker emotions
  - Identify emotional patterns and changes
  - Emotional context annotation

- **🎵 Ambient Sound Classification**
  - Detect and classify background sounds
  - Environmental context analysis
  - Noise type identification

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Conda package manager

### Installation

1. Create and activate a conda environment:
```bash
conda create -n nirva python=3.10
conda activate nirva
```

2. Install required packages in the following order:
```bash
# Basic scientific packages
conda install numpy scipy pandas

# PyTorch and torchaudio
conda install pytorch==1.13.1 torchaudio==0.13.1 -c pytorch

# Audio processing libraries
conda install -c conda-forge librosa soundfile
pip install pydub

# Transformers and dependencies
pip install transformers

# Whisper
pip install openai-whisper

# Streamlit
pip install streamlit

# Pyannote audio
pip install pyannote.audio==2.1.1
```

### Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload audio files and process them using the available features:
   - Select files to process
   - Adjust silence detection parameters if needed
   - Choose between silence trimming and full processing
   - Download processed results

## 🛠️ Technical Details

- **Frontend**: Streamlit
- **Audio Processing**: PyTorch, torchaudio, librosa
- **Speech Recognition**: Whisper
- **Speaker Diarization**: pyannote.audio
- **Visualization**: Matplotlib, Streamlit components

## 📊 Processing Pipeline

1. **File Upload**
   - Support for multiple file uploads
   - Automatic format detection
   - Initial audio analysis

2. **Silence Processing**
   - Configurable silence detection
   - Visual timeline generation
   - Non-destructive trimming

3. **Audio Analysis**
   - Speaker diarization
   - Transcription
   - Emotion and ambient sound analysis

4. **Results Presentation**
   - Interactive visualizations
   - Downloadable processed files
   - Detailed analytics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Pyannote.audio for speaker diarization
- Streamlit for the web interface 