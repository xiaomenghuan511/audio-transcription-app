# Audio Transcription Web App

This application provides a comprehensive audio transcription service with speaker diarization, emotion detection, and ambient sound classification.

## Features
- Audio file upload and processing
- Automatic transcription using Whisper
- Speaker diarization with pyannote.audio
- Emotion detection using SpeechBrain
- Ambient sound classification using YAMNet
- Timestamped output with speaker and emotion labels
- Background sound detection for non-speech segments

## Requirements
- Python 3.8 or higher
- HuggingFace account and access token
- Sufficient disk space for audio processing
- GPU recommended for faster processing (but not required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your HuggingFace token:
- Go to https://huggingface.co/settings/tokens
- Create a new token with read access
- Save it for use in the application

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. In the web interface:
- Enter your HuggingFace token in the sidebar
- Set the recording date
- Upload one or more audio files
- Wait for processing to complete
- Save the transcript using the "Save Transcript" button

## Output Format

The application generates timestamped transcripts with the following format:

```
[00:00:12]
[Speaker A], [confident]: Hello, this is a test...

[00:00:18]
[Ambient Sound - background music]
```

## Supported Audio Formats
- WAV
- MP3
- OGG
- M4A

## Notes
- Supports multiple audio files processing
- Automatically handles file format conversion
- Detects and filters out repetitive or nonsensical content
- Saves transcripts with date-based filenames

## License
MIT License

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 