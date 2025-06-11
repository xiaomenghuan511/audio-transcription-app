import streamlit as st
import os
import tempfile
from datetime import datetime
from audio_utils import AudioPreprocessor, AudioTrimmer
from transcription import AudioAnalyzer
from pydub import AudioSegment
import time
import pandas as pd
import io

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
        st.header("Settings")
        
        # Date input
        recording_date = st.date_input(
            "Recording Date",
            datetime.now().date()
        )
        
        # HuggingFace token input
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        
        # Silence detection settings
        st.subheader("Silence Detection Settings")
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
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['wav', 'mp3', 'm4a'],
            accept_multiple_files=True
        )

    # Main content area
    if not hf_token:
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
        tabs = st.tabs(["Silence Detection", "Full Processing"])
        
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

        # Full Processing Tab
        with tabs[1]:
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
                        analyzer = AudioAnalyzer(hf_token)
                        
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
            
            # Reset session state
            st.session_state.processed_files = []
            st.session_state.current_results = []
            st.session_state.current_text = ""
            st.session_state.uploaded_files_info = {}
            st.session_state.selected_files = []
            st.session_state.trimmed_files = {}
            st.session_state.processing_results = {}
            st.experimental_rerun()
    else:
        st.info("Please upload audio files in the sidebar to begin.")

if __name__ == "__main__":
    main() 