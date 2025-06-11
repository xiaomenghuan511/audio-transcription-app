import streamlit as st
import os
import tempfile
from datetime import datetime
from audio_utils import AudioPreprocessor
from transcription import AudioAnalyzer

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""

# Set page config
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ Audio Transcription App")
st.markdown("""
This app processes audio files to create detailed transcripts with:
- Speaker identification
- Emotion detection
- Ambient sound classification
""")

# Sidebar for input controls
with st.sidebar:
    st.header("Settings")
    
    # HuggingFace token input
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        help="Required for speaker diarization"
    )
    
    # Date input
    recording_date = st.date_input(
        "Recording Date",
        datetime.now().date()
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Audio Files",
        type=['wav', 'mp3', 'ogg', 'm4a'],
        accept_multiple_files=True
    )

# Main content area
if uploaded_files and hf_token:
    # Initialize processors
    preprocessor = AudioPreprocessor()
    analyzer = AudioAnalyzer(hf_token)
    
    # Process each file
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            st.write(f"Processing {uploaded_file.name}...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Pre-process audio
                progress_bar.progress(20)
                chunks = preprocessor.process_audio_file(temp_path)
                
                # Process chunks
                results = []
                for i, chunk in enumerate(chunks):
                    # Update progress
                    progress = 20 + (i + 1) / len(chunks) * 60
                    progress_bar.progress(int(progress))
                    
                    # Process chunk
                    result = analyzer.process_chunk(chunk, temp_dir)
                    results.append(result)
                
                # Format output
                progress_bar.progress(90)
                transcript = analyzer.format_output(results)
                
                # Post-process
                transcript = analyzer.post_process(transcript)
                
                # Update session state
                st.session_state.processed_files.append(uploaded_file.name)
                st.session_state.current_transcript = transcript
                
                progress_bar.progress(100)
                
    # Display transcript
    st.header("Transcript")
    st.text_area(
        "Processed Text",
        st.session_state.current_transcript,
        height=400
    )
    
    # Save button
    if st.button("Save Transcript"):
        date_str = recording_date.strftime("%Y%m%d")
        filename = f"{date_str}_transcript.txt"
        
        with open(filename, 'w') as f:
            f.write(st.session_state.current_transcript)
        
        st.success(f"Transcript saved as {filename}")
        
    # Clear button
    if st.button("Clear"):
        st.session_state.processed_files = []
        st.session_state.current_transcript = ""
        st.experimental_rerun()

else:
    if not hf_token:
        st.warning("Please enter your HuggingFace token in the sidebar.")
    if not uploaded_files:
        st.info("Please upload audio files in the sidebar to begin processing.") 