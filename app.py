import streamlit as st
import os
import tempfile
from datetime import datetime
from audio_utils import AudioPreprocessor, AudioTrimmer, AudioSegmenter
from transcription import (
    OptimizedMultiModelTranscriber, OptimizedSpeakerDiarizer, 
    WhisperCloudModel, WhisperLocalModel, DeepgramModel
)
from pydub import AudioSegment
import time
import pandas as pd
import io
import asyncio
import requests
import zipfile

# Set page config
st.set_page_config(
    page_title="Nirva Audio Lab",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = []
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = {}
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'trimmed_files' not in st.session_state:
    st.session_state.trimmed_files = {}
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}
if 'segmented_files' not in st.session_state:
    st.session_state.segmented_files = {}
if 'segmentation_params' not in st.session_state:
    st.session_state.segmentation_params = {
        'min_length': 30,
        'max_length': 60
    }
if 'segmentation_completed' not in st.session_state:
    st.session_state.segmentation_completed = False
if 'total_processing_time' not in st.session_state:
    st.session_state.total_processing_time = 0
if 'transcription_results' not in st.session_state:
    st.session_state.transcription_results = {}
if 'transcription_completed' not in st.session_state:
    st.session_state.transcription_completed = False
# Add new session state variables for API keys and model selections
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'huggingface_token' not in st.session_state:
    st.session_state.huggingface_token = ""
if 'deepgram_api_key' not in st.session_state:
    st.session_state.deepgram_api_key = ""
if 'whisper_cloud_enabled' not in st.session_state:
    st.session_state.whisper_cloud_enabled = False
if 'whisper_local_enabled' not in st.session_state:
    st.session_state.whisper_local_enabled = False
if 'deepgram_enabled' not in st.session_state:
    st.session_state.deepgram_enabled = False
if 'whisper_cloud_sizes' not in st.session_state:
    st.session_state.whisper_cloud_sizes = ["base"]
if 'whisper_local_sizes' not in st.session_state:
    st.session_state.whisper_local_sizes = ["base"]

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def get_download_link(file_path, filename):
    """Generate a download link for a file"""
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    return bytes_data

def format_file_size(size_in_bytes):
    """Convert file size to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} GB"

def format_duration(seconds):
    """Convert duration in seconds to minutes and seconds format"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02.0f}"

def format_time(seconds):
    """Format processing time in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} minutes {remaining_seconds:.1f} seconds"

def validate_openai_api(api_key: str) -> bool:
    """Validate OpenAI API key"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Test with a simple API call
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

def validate_deepgram_api(api_key: str) -> bool:
    """Validate Deepgram API key"""
    try:
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        # Test with a simple API call
        response = requests.get(
            "https://api.deepgram.com/v1/projects",
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

async def transcribe_with_realtime_updates(transcriber, segment_paths, original_audio_path,
                                         progress_bar, status_text, realtime_areas, log_area):
    """Custom transcription method with real-time updates"""
    # Perform diarization only if enabled
    full_diarization = None
    if transcriber.diarizer:
        log_area.text("Performing speaker diarization on full audio...")
        full_diarization = transcriber.diarizer.diarize_full_audio(original_audio_path)
        log_area.text("Speaker diarization completed")
    
    # Initialize results storage
    results = {model.model_name: [] for model in transcriber.models}
    accumulated_transcripts = {model.model_name: "" for model in transcriber.models}
    
    # Process segments one by one for real-time display
    total_segments = len(segment_paths)
    for i, segment_info in enumerate(segment_paths):
        # Update progress
        progress = 0.3 + (0.6 * (i + 1) / total_segments)
        progress_bar.progress(progress)
        status_text.text(f"Processing segment {i+1}/{total_segments}...")
        
        # Log segment processing
        log_area.text(f"Processing segment {i+1}/{total_segments} from {segment_info['path']}")
        
        # Transcribe this segment with all models
        segment_results = await transcriber.transcribe_segment_batch(segment_info, full_diarization)
        
        # Update results and real-time display
        for model_name, result in segment_results.items():
            results[model_name].append(result)
            
            # Update accumulated transcript
            if 'formatted_output' in result:
                accumulated_transcripts[model_name] += result['formatted_output'] + "\n"
                log_area.text(f"Updated transcript for {model_name} - Segment {i+1}")
            
            # Update real-time display
            if model_name in realtime_areas:
                realtime_areas[model_name].text_area(
                    f"Progress: {i+1}/{total_segments}",
                    accumulated_transcripts[model_name],
                    height=300,
                    key=f"realtime_{model_name}_{i}"
                )
        
        log_area.text(f"Completed processing segment {i+1}/{total_segments}")
    
    return results

def main():
    st.title("🎙️ Nirva Audio Lab")
    st.markdown("""
    This app processes audio files to create detailed transcripts with:
    - ✂️ Detect and trim silence 
    - 🔄 Audio segmentation
    - 🗣️ Transcript with Speaker identification
    - 😊 Emotion detection
    - 🎵 Ambient sound classification
    """)

    # Sidebar
    with st.sidebar:
        st.header("Audio Settings")
        
        # Date input
        recording_date = st.date_input(
            "Recording Date",
            datetime.now().date()
        )
        
        # Multiple file uploader - moved to top
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['wav', 'mp3', 'm4a'],
            accept_multiple_files=True
        )
        
        # API Keys
        st.subheader("🔑 API Keys")
        
        # Initialize validation variables
        openai_api_valid = False
        deepgram_api_valid = False
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Required for Whisper Cloud API"
        )
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            if st.button("🔍 Validate OpenAI API", key="validate_openai"):
                with st.spinner("Validating OpenAI API..."):
                    openai_api_valid = validate_openai_api(openai_api_key)
                    if openai_api_valid:
                        st.success("✅ OpenAI API key is valid!")
                    else:
                        st.error("❌ OpenAI API key is invalid or unreachable!")
        
        # HuggingFace Token
        huggingface_token = st.text_input(
            "HuggingFace Token",
            type="password",
            value=st.session_state.huggingface_token,
            help="Required for downloading Whisper models and speaker diarization"
        )
        if huggingface_token:
            st.session_state.huggingface_token = huggingface_token
            if st.button("🔍 Validate HuggingFace Token", key="validate_huggingface"):
                with st.spinner("Validating HuggingFace token..."):
                    try:
                        # First check if token is valid by checking user info
                        headers = {"Authorization": f"Bearer {huggingface_token}"}
                        
                        # Try to access the model directly
                        model_response = requests.get(
                            "https://huggingface.co/pyannote/speaker-diarization",
                            headers=headers,
                            timeout=5
                        )
                        
                        if model_response.status_code == 200:
                            st.success("✅ HuggingFace token is valid and has access to the diarization model!")
                        elif model_response.status_code == 403:
                            st.warning("⚠️ Please accept the user conditions at https://huggingface.co/pyannote/speaker-diarization")
                        else:
                            # Try to access the API as a fallback
                            api_response = requests.get(
                                "https://huggingface.co/api/whoami",
                                headers=headers,
                                timeout=5
                            )
                            
                            if api_response.status_code == 200:
                                st.success("✅ HuggingFace token is valid!")
                            else:
                                st.error(f"❌ Token validation failed. Status code: {api_response.status_code}")
                                st.error("Please make sure you have:")
                                st.error("1. Created a token with 'read' access")
                                st.error("2. Accepted the user agreement")
                                st.error("3. Accepted the model terms at https://huggingface.co/pyannote/speaker-diarization")
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Network error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error validating token: {str(e)}")
        
        # Deepgram API Key
        deepgram_api_key = st.text_input(
            "Deepgram API Key",
            type="password",
            value=st.session_state.deepgram_api_key,
            help="Required for Deepgram API"
        )
        if deepgram_api_key:
            st.session_state.deepgram_api_key = deepgram_api_key
            if st.button("🔍 Validate Deepgram API", key="validate_deepgram"):
                with st.spinner("Validating Deepgram API..."):
                    deepgram_api_valid = validate_deepgram_api(deepgram_api_key)
                    if deepgram_api_valid:
                        st.success("✅ Deepgram API key is valid!")
                    else:
                        st.error("❌ Deepgram API key is invalid or unreachable!")
        
        # Silence detection settings
        st.subheader("✂️ Silence Detection Settings")
        min_silence_len = st.slider(
            "Minimum Silence Length (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Minimum length of silence to be detected"
        )
        silence_thresh = st.slider(
            "Silence Threshold (dB)",
            min_value=-60,
            max_value=-20,
            value=-40,
            step=1,
            help="Sound levels below this threshold are considered silence"
        )
        
        # Audio segmentation settings
        st.subheader("🔄 Audio Segmentation Settings")
        min_segment_length = st.slider(
            "Minimum Segment Length (seconds)",
            min_value=20,
            max_value=45,
            value=st.session_state.segmentation_params['min_length'],
            step=5,
            help="Minimum length of each audio segment"
        )
        max_segment_length = st.slider(
            "Maximum Segment Length (seconds)",
            min_value=45,
            max_value=90,
            value=st.session_state.segmentation_params['max_length'],
            step=5,
            help="Maximum length of each audio segment"
        )
        
        # Update segmentation parameters in session state
        st.session_state.segmentation_params['min_length'] = min_segment_length
        st.session_state.segmentation_params['max_length'] = max_segment_length
        
        # Model Selection
        st.subheader("🎯 Select Transcription Models")
        
        # Whisper Cloud (single checkbox)
        whisper_cloud_enabled = st.checkbox("Whisper Cloud (OpenAI's hosted model)", value=True)
        
        # Whisper Local (with size selection)
        whisper_local_enabled = st.checkbox("Whisper Local")
        if whisper_local_enabled:
            whisper_local_models = ["tiny", "base", "small"]
            selected_local_model = st.selectbox(
                "Select Whisper Local Model Size",
                whisper_local_models,
                index=1  # Default to "base"
            )
        
        # WhisperX (with size selection)
        whisperx_enabled = st.checkbox("WhisperX")
        if whisperx_enabled:
            whisperx_models = ["tiny", "base", "small", "medium", "large"]
            selected_whisperx_model = st.selectbox(
                "Select WhisperX Model Size",
                whisperx_models,
                index=1  # Default to "base"
            )
        
        # Collect selected models
        selected_models = []
        if whisper_cloud_enabled:
            selected_models.append("Whisper Cloud")
        if whisper_local_enabled:
            selected_models.append("Whisper Local")
        if whisperx_enabled:
            selected_models.append("WhisperX")
        
        if not selected_models:
            st.error("Please select at least one model")
            st.stop()
        
        # Transcription settings
        st.subheader("🗣️ Transcription with Speaker Diarization")
        
        # Diarization toggle
        enable_diarization = st.checkbox(
            "Enable Speaker Diarization",
            value=True,
            help="Enable speaker diarization to identify different speakers in the audio"
        )
        
        # Initialize diarizer if enabled
        diarizer = None
        if enable_diarization:
            if not huggingface_token:
                st.error("HuggingFace token is required for speaker diarization")
                st.stop()
            diarizer = OptimizedSpeakerDiarizer(huggingface_token)

    # Main content area
    if not huggingface_token:
        st.warning("Please enter your HuggingFace token in the sidebar to continue.")
        st.stop()

    if uploaded_files:
        # Process uploaded files information
        files_info = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files_info:
                # Save file and get information
                temp_path = save_uploaded_file(uploaded_file)
                try:
                    audio = AudioSegment.from_file(temp_path)
                    duration = len(audio) / 1000.0  # Convert to seconds
                    st.session_state.uploaded_files_info[uploaded_file.name] = {
                        'path': temp_path,
                        'size': uploaded_file.size,
                        'duration': duration,
                        'processed': False,
                        'trimmed': False
                    }
                except Exception as e:
                    st.error(f"Error loading audio file {uploaded_file.name}: {str(e)}")
                    continue

            # Prepare information for display
            file_info = st.session_state.uploaded_files_info[uploaded_file.name]
            files_info.append({
                'File Name': uploaded_file.name,
                'Size': format_file_size(file_info['size']),
                'Duration': format_duration(file_info['duration']),
                'Status': '✅ Processed' if file_info['processed'] else '⏳ Pending',
                'Trimmed': '✂️ Yes' if file_info['trimmed'] else '❌ No'
            })

        # Display files information in a table
        st.subheader("📁 Uploaded Files")
        files_df = pd.DataFrame(files_info)
        st.dataframe(
            files_df,
            use_container_width=True,
            hide_index=True
        )

        # File selection
        st.session_state.selected_files = st.multiselect(
            "Select files to process",
            [file.name for file in uploaded_files],
            default=[file.name for file in uploaded_files]
        )

        # Add tabs for different processing steps
        tabs = st.tabs(["Silence Detection", "Audio Segmentation", "Transcription & Diarization", "Full Processing"])
        
        # Silence Detection Tab
        with tabs[0]:
            if st.button("🔍 Detect and Trim Silence", type="primary"):
                if not st.session_state.selected_files:
                    st.warning("Please select at least one file to process.")
                    st.stop()

                # Create a progress container
                progress_container = st.empty()
                
                total_files = len(st.session_state.selected_files)
                total_start_time = time.time()
                
                for idx, file_name in enumerate(st.session_state.selected_files):
                    file_info = st.session_state.uploaded_files_info[file_name]
                    
                    # Update progress
                    with progress_container.container():
                        st.subheader(f"Processing: {file_name} ({idx + 1}/{total_files})")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    # Create trimmer instance
                    start_time = time.time()
                    trimmer = AudioTrimmer(file_info['path'])
                    
                    # Update progress - 20%
                    progress_bar.progress(0.2)
                    status_text.text("Detecting silence...")
                    
                    # Detect silence
                    results = trimmer.detect_silence(
                        min_silence_len=int(min_silence_len * 1000),
                        silence_thresh=silence_thresh
                    )
                    
                    # Update progress - 50%
                    progress_bar.progress(0.5)
                    status_text.text("Creating visualization...")
                    
                    # Create visualization
                    timeline_image = trimmer.create_timeline_visualization()
                    
                    # Update progress - 80%
                    progress_bar.progress(0.8)
                    status_text.text("Preparing trimmed audio...")
                    
                    # Get trimmed audio
                    trimmed_audio = trimmer.get_trimmed_audio()
                    trimmed_path = os.path.join(
                        os.path.dirname(file_info['path']),
                        f"trimmed_{file_name}"
                    )
                    trimmed_audio.export(trimmed_path, format=os.path.splitext(file_name)[1][1:])
                    
                    # Update file info
                    st.session_state.uploaded_files_info[file_name]['trimmed'] = True
                    st.session_state.trimmed_files[file_name] = {
                        'path': trimmed_path,
                        'duration': results['trimmed_duration']
                    }
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Store results in session state
                    st.session_state.processing_results[file_name] = {
                        'results': results,
                        'timeline_image': timeline_image,
                        'processing_time': processing_time,
                        'trimmed_path': trimmed_path
                    }
                    
                    # Update progress - 100%
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                
                # Display total processing time
                total_processing_time = time.time() - total_start_time
                st.success(f"Total processing time for {total_files} files: {format_time(total_processing_time)}")
                
                # Clear progress container
                progress_container.empty()
            
            # Display all stored results
            if st.session_state.processing_results:
                st.subheader("📊 Processing Results")
                for file_name, result_data in st.session_state.processing_results.items():
                    with st.expander(f"Results for: {file_name}", expanded=True):
                        # Display processing time
                        st.info(f"Processing time: {format_time(result_data['processing_time'])}")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Duration", 
                                    f"{format_duration(result_data['results']['original_duration'])}")
                        with col2:
                            st.metric("Trimmed Duration", 
                                    f"{format_duration(result_data['results']['trimmed_duration'])}")
                        with col3:
                            st.metric("Trim Ratio", 
                                    f"{result_data['results']['trim_ratio']:.1f}%")
                        
                        # Display timeline visualization
                        st.image(f"data:image/png;base64,{result_data['timeline_image']}")
                        
                        # Create download button
                        if os.path.exists(result_data['trimmed_path']):
                            download_data = get_download_link(result_data['trimmed_path'], f"trimmed_{file_name}")
                            st.download_button(
                                label=f"💾 Download Trimmed Audio",
                                data=download_data,
                                file_name=f"trimmed_{file_name}",
                                mime=f"audio/{os.path.splitext(file_name)[1][1:]}"
                            )
                        
                        st.divider()

        # Audio Segmentation Tab
        with tabs[1]:
            st.subheader("🔄 Audio Segmentation")
            
            # Check if there are trimmed files available
            available_files = [
                file_name for file_name, info in st.session_state.trimmed_files.items()
                if os.path.exists(info['path'])
            ]
            
            if not available_files:
                st.warning("Please process some files in the Silence Detection tab first.")
            else:
                # File selection for segmentation
                files_to_segment = st.multiselect(
                    "Select trimmed files to segment",
                    available_files,
                    default=available_files
                )
                
                # Display previous results if available
                if st.session_state.segmentation_completed:
                    st.success(f"Segmentation completed for {len(st.session_state.segmented_files)} files!")
                    st.info(f"Total processing time: {format_time(st.session_state.total_processing_time)}")
                    
                    # Display results for all processed files
                    for file_name, seg_info in st.session_state.segmented_files.items():
                        with st.expander(f"Results for: {file_name}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Segments", str(seg_info['segment_count']))
                            with col2:
                                avg_duration = sum(seg['duration'] for seg in seg_info['segments']) / len(seg_info['segments'])
                                st.metric("Average Segment Duration", f"{avg_duration:.1f}s")
                            with col3:
                                st.metric("Original Duration", f"{st.session_state.trimmed_files[file_name]['duration']:.1f}s")
                            
                            # Display segmentation statistics
                            with st.expander("📊 Segmentation Statistics", expanded=True):
                                if seg_info['segments']:
                                    # Calculate statistics
                                    total_duration = sum(seg['duration'] for seg in seg_info['segments'])
                                    avg_duration = total_duration / len(seg_info['segments'])
                                    min_duration = min(seg['duration'] for seg in seg_info['segments'])
                                    max_duration = max(seg['duration'] for seg in seg_info['segments'])
                                    
                                    # Display statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Segments", len(seg_info['segments']))
                                    with col2:
                                        st.metric("Average Duration", f"{avg_duration:.1f}s")
                                    with col3:
                                        st.metric("Min Duration", f"{min_duration:.1f}s")
                                    with col4:
                                        st.metric("Max Duration", f"{max_duration:.1f}s")
                                    
                                    # Display segment timeline
                                    st.write("Segment Timeline:")
                                    timeline_data = []
                                    for seg in seg_info['segments']:
                                        timeline_data.append({
                                            'Start': seg['start'],
                                            'End': seg['end'],
                                            'Duration': seg['duration']
                                        })
                                    st.dataframe(timeline_data)
                                else:
                                    st.info("Audio file is too short to be segmented. It will be processed as a single segment.")
                
                # Process button
                if st.button("🔄 Process Selected Files", type="primary"):
                    if not files_to_segment:
                        st.warning("Please select at least one file to process.")
                        st.stop()
                    
                    # Create containers for real-time display
                    progress_container = st.empty()
                    realtime_container = st.empty()
                    
                    # Process each file
                    for file_name in files_to_segment:
                        st.write(f"Starting processing for {file_name}...")
                        try:
                            with progress_container.container():
                                st.subheader(f"Processing: {file_name}")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Create a log container that will persist
                                log_container = st.container()
                                with log_container:
                                    st.subheader("📋 Processing Log")
                                    log_area = st.empty()
                                    log_area.text("Starting audio processing...")
                                
                                # Get audio file path
                                audio_path = st.session_state.trimmed_files[file_name]['path']
                                audio_duration = st.session_state.trimmed_files[file_name]['duration']
                                
                                # Check if audio is too short for segmentation
                                if audio_duration < min_segment_length:
                                    log_area.text("Audio file is too short for segmentation. Processing as a single segment...")
                                    # Create a single segment for the entire audio
                                    segment_info = {
                                        'path': audio_path,
                                        'start': 0,
                                        'end': audio_duration,
                                        'duration': audio_duration,
                                        'index': 0,
                                        'audio': AudioSegment.from_file(audio_path)  # Add audio object
                                    }
                                    segments = [segment_info]
                                    
                                    # Create a zip file with the single segment
                                    zip_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(file_name)[0]}_segments.zip")
                                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                                        temp_path = os.path.join(tempfile.gettempdir(), "segment_1.wav")
                                        segment_info['audio'].export(temp_path, format='wav')
                                        zipf.write(temp_path, "segment_1.wav")
                                        os.remove(temp_path)  # Clean up temp file
                                    
                                    # Update session state
                                    st.session_state.segmented_files[file_name] = {
                                        'segments': segments,
                                        'zip_path': zip_path
                                    }
                                    
                                    log_area.text("Processing completed successfully!")
                                    progress_bar.progress(1.0)
                                    continue
                                
                                # Normal segmentation process for longer audio files
                                log_area.text("Initializing audio segmenter...")
                                segmenter = AudioSegmenter(
                                    audio_path=audio_path,
                                    min_segment_length=min_segment_length,
                                    max_segment_length=max_segment_length
                                )
                                
                                # Process the file
                                log_area.text("Starting segmentation...")
                                segments = segmenter.segment_audio()
                                
                                if not segments:
                                    st.error(f"Failed to segment {file_name}")
                                    continue
                                
                                # Create zip file with segments
                                log_area.text("Creating zip file with segments...")
                                zip_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(file_name)[0]}_segments.zip")
                                with zipfile.ZipFile(zip_path, 'w') as zipf:
                                    for i, segment in enumerate(segments):
                                        # Handle both segmented and non-segmented audio
                                        if 'audio' in segment:
                                            # For normal segmented audio
                                            temp_path = os.path.join(tempfile.gettempdir(), f"segment_{i+1}.wav")
                                            segment['audio'].export(temp_path, format='wav')
                                            zipf.write(temp_path, f"segment_{i+1}.wav")
                                            os.remove(temp_path)  # Clean up temp file
                                        else:
                                            # For short audio that wasn't segmented
                                            zipf.write(segment['path'], f"segment_{i+1}.wav")
                                
                                # Update session state
                                st.session_state.segmented_files[file_name] = {
                                    'segments': segments,
                                    'zip_path': zip_path
                                }
                                
                                log_area.text("Processing completed successfully!")
                                progress_bar.progress(1.0)
                                
                        except Exception as e:
                            st.error(f"Error processing {file_name}: {str(e)}")
                            continue

        # Transcription & Diarization Tab
        with tabs[2]:
            st.subheader("🗣️ Transcription with Speaker Diarization")
            
            # Check if there are segmented files available
            available_segmented_files = [
                file_name for file_name, info in st.session_state.segmented_files.items()
                if os.path.exists(info['zip_path'])
            ]
            
            if not available_segmented_files:
                st.warning("Please process some files in the Audio Segmentation tab first.")
            else:
                # Validate model selection
                selected_models = []
                if whisper_cloud_enabled:
                    selected_models.append("Whisper Cloud")
                
                if whisper_local_enabled:
                    selected_models.append("Whisper Local")
                
                if whisperx_enabled:
                    selected_models.append("WhisperX")
                
                if len(selected_models) > 3:
                    st.error("Please select at most 3 transcription models in total.")
                    st.stop()
                elif len(selected_models) == 0:
                    st.warning("Please select at least one transcription model in the sidebar.")
                    st.stop()
                
                # Display selected models
                st.info(f"Selected models: {', '.join(selected_models)}")
                
                # File selection for transcription
                files_to_transcribe = st.multiselect(
                    "Select segmented files to transcribe",
                    available_segmented_files,
                    default=available_segmented_files[:1]
                )
                
                # Display previous results if available
                if st.session_state.transcription_completed and st.session_state.transcription_results:
                    st.success("Transcription completed!")
                    
                    for file_name, model_results in st.session_state.transcription_results.items():
                        st.subheader(f"Results for: {file_name}")
                        
                        # Create columns for each model
                        if len(model_results) == 1:
                            cols = [st.container()]
                        elif len(model_results) == 2:
                            cols = st.columns(2)
                        else:
                            cols = st.columns(3)
                        
                        for i, (model_name, transcript) in enumerate(model_results.items()):
                            with cols[i]:
                                st.write(f"**{model_name}**")
                                st.text_area(
                                    f"Transcript ({model_name})",
                                    transcript,
                                    height=400,
                                    key=f"transcript_{file_name}_{model_name}"
                                )
                                
                                # Download button
                                st.download_button(
                                    label=f"📥 Download {model_name}",
                                    data=transcript,
                                    file_name=f"{os.path.splitext(file_name)[0]}_{model_name}_transcript.txt",
                                    mime="text/plain"
                                )
                
                # Process button
                if st.button("🗣️ Transcribe Selected Files", type="primary"):
                    if not files_to_transcribe:
                        st.warning("Please select at least one file to transcribe.")
                        st.stop()
                    
                    # Create containers for real-time display
                    progress_container = st.empty()
                    realtime_container = st.empty()
                    metrics_container = st.empty()
                    log_container = st.container()
                    
                    # Initialize log area
                    with log_container:
                        st.subheader("📋 Processing Log")
                        log_area = st.empty()
                    
                    # Initialize selected models
                    models = []
                    log_area.text("Initializing models...")
                    if "Whisper Cloud" in selected_models:
                        log_area.text("Initializing Whisper Cloud model...")
                        try:
                            if not openai_api_key:
                                st.error("OpenAI API key is required for Whisper Cloud")
                                st.stop()
                            model = WhisperCloudModel("whisper-1", openai_api_key)
                            log_area.text("Successfully initialized Whisper Cloud model")
                            models.append(model)
                        except Exception as e:
                            st.error(f"Error initializing Whisper Cloud model: {str(e)}")
                            st.stop()
                    
                    if "Whisper Local" in selected_models:
                        log_area.text("Initializing Whisper Local model...")
                        try:
                            if not selected_local_model:
                                st.error("Please select a local model size")
                                st.stop()
                            # 创建模型实例但不立即加载
                            model = WhisperLocalModel(selected_local_model)
                            # 尝试加载模型
                            try:
                                model._load_model()
                                log_area.text(f"Successfully initialized Whisper Local {selected_local_model} model")
                                models.append(model)
                            except Exception as e:
                                st.error(f"Failed to load Whisper Local model: {str(e)}")
                                st.stop()
                        except Exception as e:
                            st.error(f"Error creating Whisper Local model: {str(e)}")
                            st.stop()
                    
                    if "WhisperX" in selected_models:
                        log_area.text("Initializing WhisperX model...")
                        try:
                            if not selected_whisperx_model:
                                st.error("Please select a WhisperX model size")
                                st.stop()
                            model = WhisperXModel(selected_whisperx_model)
                            log_area.text(f"Successfully initialized WhisperX {selected_whisperx_model} model")
                            models.append(model)
                        except Exception as e:
                            st.error(f"Error initializing WhisperX model: {str(e)}")
                            st.stop()
                    
                    if not models:
                        st.error("No models were initialized. Please select at least one model.")
                        st.stop()
                    
                    # Initialize diarizer only if enabled
                    diarizer = None
                    if enable_diarization:
                        if not huggingface_token:
                            st.error("HuggingFace token is required for speaker diarization")
                            st.stop()
                        diarizer = OptimizedSpeakerDiarizer(huggingface_token)
                    
                    # Create transcriber
                    st.write("Initializing transcriber...")
                    try:
                        transcriber = OptimizedMultiModelTranscriber(models, diarizer)
                        st.write("Successfully initialized transcriber")
                    except Exception as e:
                        st.error(f"Error initializing transcriber: {str(e)}")
                        st.stop()
                    
                    # Process each file
                    for file_name in files_to_transcribe:
                        st.write(f"Starting processing for {file_name}...")
                        try:
                            with progress_container.container():
                                st.subheader(f"Transcribing: {file_name}")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Create a log container that will persist
                                log_container = st.container()
                                with log_container:
                                    st.subheader("📋 Processing Log")
                                    log_area = st.empty()
                                    log_area.text("Starting transcription process...")
                                
                                # Real-time results display
                                with realtime_container.container():
                                    st.subheader("📝 Real-time Transcription Results")
                                    
                                    # Create columns for real-time display
                                    if len(models) == 1:
                                        realtime_cols = [st.container()]
                                    elif len(models) == 2:
                                        realtime_cols = st.columns(2)
                                    else:
                                        realtime_cols = st.columns(3)
                                    
                                    # Initialize real-time display areas
                                    realtime_areas = {}
                                    for i, model in enumerate(models):
                                        with realtime_cols[i]:
                                            st.write(f"**{model.model_name}**")
                                            realtime_areas[model.model_name] = st.empty()
                                
                                # Get segments and original audio path
                                log_area.text("Loading audio segments...")
                                segments_info = st.session_state.segmented_files[file_name]['segments']
                                original_audio_path = st.session_state.trimmed_files[file_name]['path']
                                log_area.text(f"Found {len(segments_info)} segments to process")
                                
                                # Prepare segment info for transcription
                                log_area.text("Preparing segments for transcription...")
                                segment_paths = []
                                for i, segment in enumerate(segments_info):
                                    temp_path = os.path.join(
                                        tempfile.gettempdir(),
                                        f"segment_{file_name}_{i}.wav"
                                    )
                                    segment['audio'].export(temp_path, format='wav')
                                    segment_paths.append({
                                        'path': temp_path,
                                        'index': i,
                                        'start': segment['start'],
                                        'end': segment['end']
                                    })
                                    log_area.text(f"Prepared segment {i+1}/{len(segments_info)}")
                                
                                # Update progress
                                progress_bar.progress(0.1)
                                status_text.text("Initializing models and diarization...")
                                
                                # Run optimized transcription with real-time updates
                                try:
                                    progress_bar.progress(0.3)
                                    status_text.text("Starting transcription process...")
                                    
                                    # Custom transcription with real-time updates
                                    log_area.text("Starting transcription process...")
                                    results = asyncio.run(
                                        transcribe_with_realtime_updates(
                                            transcriber, segment_paths, original_audio_path,
                                            progress_bar, status_text, realtime_areas, log_area
                                        )
                                    )
                                    
                                    # Update progress
                                    progress_bar.progress(0.9)
                                    status_text.text("Combining results...")
                                    
                                    # Combine results
                                    st.write("Combining transcription results...")
                                    final_transcripts = transcriber.combine_results(results)
                                    
                                    # Store results
                                    st.session_state.transcription_results[file_name] = final_transcripts
                                    
                                    # Clean up temporary files
                                    for segment_path in segment_paths:
                                        try:
                                            os.remove(segment_path['path'])
                                        except Exception as e:
                                            st.warning(f"Could not remove temporary file {segment_path['path']}: {str(e)}")
                                    
                                    # Update progress
                                    progress_bar.progress(1.0)
                                    status_text.text("Transcription completed!")
                                    
                                    # Display metrics at the top
                                    st.write("### 📊 Transcription Metrics")
                                    metrics_cols = st.columns(len(models))
                                    for i, model in enumerate(models):
                                        metrics = model.metrics
                                        with metrics_cols[i]:
                                            st.write(f"**{model.model_name}**")
                                            st.metric("Transcription Time", f"{metrics['transcription_time']:.2f}s")
                                            st.metric("Input Tokens", f"{metrics['input_tokens']}")
                                            st.metric("Output Tokens", f"{metrics['output_tokens']}")
                                            if metrics['cost'] > 0:
                                                st.metric("Cost", f"${metrics['cost']:.4f}")
                                            else:
                                                st.metric("Cost", "Free (Local Model)")
                                    
                                    st.divider()
                                    
                                    # Display results
                                    st.write("### 📝 Final Transcription Results")
                                    for model_name, transcript in final_transcripts.items():
                                        with st.expander(f"Results from {model_name}"):
                                            st.text_area("Transcription", transcript, height=300)
                                    
                                except Exception as e:
                                    st.error(f"Error during transcription: {str(e)}")
                                    st.error("Detailed error information:")
                                    st.exception(e)
                                    st.stop()
                        
                        except Exception as e:
                            st.error(f"Error processing file {file_name}: {str(e)}")
                            st.error("Detailed error information:")
                            st.exception(e)
                            st.stop()
                    
                    # Mark as completed and rerun
                    st.session_state.transcription_completed = True
                    progress_container.empty()
                    realtime_container.empty()
                    metrics_container.empty()
                    st.rerun()

        # Full Processing Tab
        with tabs[3]:
            if st.button("🎯 Process Selected Audio Files", type="primary"):
                if not st.session_state.selected_files:
                    st.warning("Please select at least one file to process.")
                    st.stop()

                for file_name in st.session_state.selected_files:
                    file_info = st.session_state.uploaded_files_info[file_name]
                    if file_info['processed']:
                        continue

                    st.subheader(f"Processing: {file_name}")
                    
                    # Use trimmed version if available
                    audio_path = (st.session_state.trimmed_files.get(file_name, {}).get('path') 
                                or file_info['path'])
                    
                    try:
                        # Initialize audio analyzer
                        analyzer = AudioAnalyzer(huggingface_token)
                        
                        # Create a temporary directory for processing
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Load audio
                            try:
                                audio = AudioSegment.from_file(audio_path)
                            except Exception as e:
                                st.error(f"Error loading audio file: {str(e)}")
                                continue

                            # Initialize progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            transcript_container = st.empty()
                            
                            # Process in chunks
                            chunk_size = 30000  # 30 seconds
                            chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
                            results = []
                            
                            for i, chunk in enumerate(chunks):
                                # Update progress
                                progress = (i + 1) / len(chunks)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing chunk {i+1} of {len(chunks)}...")
                                
                                # Process chunk
                                chunk_info = {
                                    'audio_data': chunk,
                                    'start_time': i * chunk_size / 1000.0,  # Convert to seconds
                                    'duration': len(chunk) / 1000.0  # pydub works in milliseconds
                                }
                                result = analyzer.process_chunk(chunk_info, temp_dir)
                                results.append(result)
                                
                                # Update display
                                formatted_text = analyzer.format_output(results)
                                transcript_container.text_area(
                                    "Transcript:",
                                    formatted_text,
                                    height=400
                                )
                                
                                # Small delay to prevent UI freezing
                                time.sleep(0.1)
                            
                            # Final cleanup
                            status_text.text("Processing complete!")
                            progress_bar.progress(1.0)
                            
                            # Post-process and display final result
                            final_text = analyzer.post_process(formatted_text)
                            transcript_container.text_area(
                                "Transcript:",
                                final_text,
                                height=400
                            )
                            
                            # Mark file as processed
                            st.session_state.uploaded_files_info[file_name]['processed'] = True
                            
                            # Save transcript
                            date_str = recording_date.strftime("%Y%m%d")
                            filename = f"{date_str}_{os.path.splitext(file_name)[0]}_transcript.txt"
                            
                            with open(filename, 'w') as f:
                                f.write(final_text)
                            
                            st.success(f"Transcript saved as {filename}")

                    except Exception as e:
                        st.error(f"Error processing audio file {file_name}: {str(e)}")

        # Clear button
        if st.button("Clear All"):
            # Remove temporary files
            for file_info in st.session_state.uploaded_files_info.values():
                try:
                    os.remove(file_info['path'])
                except:
                    pass
            
            for trimmed_info in st.session_state.trimmed_files.values():
                try:
                    os.remove(trimmed_info['path'])
                except:
                    pass
            
            for segmented_info in st.session_state.segmented_files.values():
                try:
                    os.remove(segmented_info['zip_path'])
                except:
                    pass
            
            # Reset session state
            st.session_state.processed_files = []
            st.session_state.current_results = []
            st.session_state.current_text = ""
            st.session_state.uploaded_files_info = {}
            st.session_state.selected_files = []
            st.session_state.trimmed_files = {}
            st.session_state.processing_results = {}
            st.session_state.segmented_files = {}
            st.session_state.segmentation_params = {
                'min_length': 30,
                'max_length': 60
            }
            st.session_state.segmentation_completed = False
            st.session_state.total_processing_time = 0
            st.session_state.transcription_results = {}
            st.session_state.transcription_completed = False
            st.session_state.openai_api_key = ""
            st.session_state.huggingface_token = ""
            st.session_state.deepgram_api_key = ""
            st.session_state.whisper_cloud_enabled = False
            st.session_state.whisper_local_enabled = False
            st.session_state.deepgram_enabled = False
            st.session_state.whisper_cloud_sizes = ["base"]
            st.session_state.whisper_local_sizes = ["base"]
            st.rerun()
    else:
        st.info("Please upload audio files in the sidebar to begin.")

if __name__ == "__main__":
    main() 