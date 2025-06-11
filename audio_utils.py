import os
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf
from typing import List, Tuple, Dict
from pydub.silence import detect_nonsilent
import matplotlib.pyplot as plt
import io
import base64
import zipfile

class AudioPreprocessor:
    def __init__(self):
        self.min_chunk_length = 30  # seconds
        self.max_chunk_length = 300  # seconds
        self.silence_threshold = -40  # dB
        self.min_silence_length = 1000  # ms

    def convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format."""
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.splitext(audio_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path

    def detect_silence(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """Detect silent segments in audio."""
        silence_ranges = []
        current_start = None
        
        for i, chunk in enumerate(audio[::100]):  # Check every 100ms
            if chunk.dBFS < self.silence_threshold:
                if current_start is None:
                    current_start = i * 100
            elif current_start is not None:
                if (i * 100 - current_start) >= self.min_silence_length:
                    silence_ranges.append((current_start, i * 100))
                current_start = None
                
        return silence_ranges

    def split_on_silence(self, audio: AudioSegment) -> List[Tuple[AudioSegment, bool]]:
        """Split audio on silence and mark speech/non-speech segments."""
        silence_ranges = self.detect_silence(audio)
        chunks = []
        last_end = 0
        
        for start, end in silence_ranges:
            if start - last_end > self.min_chunk_length * 1000:  # Convert to ms
                chunk = audio[last_end:start]
                chunks.append((chunk, True))  # True indicates speech
            
            if end - start > self.min_silence_length:
                silence_chunk = audio[start:end]
                chunks.append((silence_chunk, False))  # False indicates silence/ambient
            
            last_end = end
            
        # Add the final chunk if needed
        if last_end < len(audio):
            final_chunk = audio[last_end:]
            chunks.append((final_chunk, True))
            
        return chunks

    def process_audio_file(self, file_path: str) -> List[Dict]:
        """Main preprocessing function."""
        # Convert to WAV if needed
        if not file_path.lower().endswith('.wav'):
            file_path = self.convert_to_wav(file_path)
            
        # Load audio
        audio = AudioSegment.from_wav(file_path)
        
        # Split into chunks based on silence
        chunks = self.split_on_silence(audio)
        
        # Prepare chunk information
        processed_chunks = []
        current_time = 0
        
        for chunk, has_speech in chunks:
            duration = len(chunk) / 1000.0  # Convert to seconds
            chunk_info = {
                'start_time': current_time,
                'duration': duration,
                'has_speech': has_speech,
                'audio_data': chunk
            }
            processed_chunks.append(chunk_info)
            current_time += duration
            
        return processed_chunks

    def save_chunk(self, chunk: AudioSegment, output_path: str):
        """Save an audio chunk to file."""
        chunk.export(output_path, format='wav')

class AudioTrimmer:
    def __init__(self, audio_path):
        self.audio = AudioSegment.from_file(audio_path)
        self.original_duration = len(self.audio) / 1000.0  # seconds
        self.silence_ranges = []
        self.non_silence_ranges = []
        self.trimmed_duration = 0
        
    def detect_silence(self, min_silence_len=1000, silence_thresh=-40):
        """
        Detect silence in audio file
        min_silence_len: minimum length of silence in ms
        silence_thresh: silence threshold in dB
        """
        # Convert to ms
        audio_len = len(self.audio)
        
        # Detect non-silent ranges
        non_silent_ranges = detect_nonsilent(
            self.audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        # Convert ranges to seconds
        self.non_silence_ranges = [(start/1000, end/1000) for start, end in non_silent_ranges]
        
        # Calculate silence ranges
        self.silence_ranges = []
        current_pos = 0
        
        for start, end in self.non_silence_ranges:
            if start * 1000 > current_pos:
                self.silence_ranges.append((current_pos/1000, start))
            current_pos = end * 1000
            
        if current_pos < audio_len:
            self.silence_ranges.append((current_pos/1000, audio_len/1000))
            
        # Calculate trimmed duration
        self.trimmed_duration = sum(end - start for start, end in self.non_silence_ranges)
        
        return {
            'original_duration': self.original_duration,
            'trimmed_duration': self.trimmed_duration,
            'trim_ratio': (self.original_duration - self.trimmed_duration) / self.original_duration * 100
        }
    
    def create_timeline_visualization(self):
        """
        Create a visualization of silence vs non-silence segments
        Returns: base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(15, 2))
        
        # Plot silence ranges in gray
        for start, end in self.silence_ranges:
            ax.axvspan(start, end, color='gray', alpha=0.3)
            
        # Plot non-silence ranges in blue
        for start, end in self.non_silence_ranges:
            ax.axvspan(start, end, color='blue', alpha=0.3)
            
        # Set axis limits
        ax.set_xlim(0, self.original_duration)
        
        # Remove y-axis
        ax.set_yticks([])
        
        # Add timeline markers
        ax.set_xticks(np.arange(0, self.original_duration + 60, 60))
        ax.set_xticklabels([f"{int(x/60):02d}:{int(x%60):02d}" for x in np.arange(0, self.original_duration + 60, 60)])
        
        # Add labels
        ax.set_xlabel("Timeline (MM:SS)")
        ax.set_title("Audio Timeline (Gray: Silence, Blue: Active Audio)")
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode to base64
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    def get_trimmed_audio(self):
        """
        Return a new AudioSegment with only non-silent parts
        """
        if not self.non_silence_ranges:
            return self.audio
            
        trimmed_audio = AudioSegment.empty()
        for start, end in self.non_silence_ranges:
            trimmed_audio += self.audio[int(start * 1000):int(end * 1000)]
            
        return trimmed_audio 

class AudioSegmenter:
    def __init__(self, audio_path: str, min_segment_length: int = 30, max_segment_length: int = 60):
        """
        Initialize AudioSegmenter
        audio_path: path to audio file
        min_segment_length: minimum segment length in seconds
        max_segment_length: maximum segment length in seconds
        """
        self.audio = AudioSegment.from_file(audio_path)
        self.min_segment_length = min_segment_length * 1000  # convert to ms
        self.max_segment_length = max_segment_length * 1000  # convert to ms
        self.segments = []
        self.original_duration = len(self.audio) / 1000.0  # seconds
        
    def find_sentence_boundaries(self, chunk_size: int = 10000) -> List[int]:
        """
        Find potential sentence boundaries using silence detection
        Returns list of timestamps in milliseconds
        """
        # Convert audio to numpy array for processing
        samples = np.array(self.audio.get_array_of_samples())
        sample_rate = self.audio.frame_rate
        
        # Use librosa to detect speech onset
        onset_frames = librosa.onset.onset_detect(
            y=samples.astype(float),
            sr=sample_rate,
            units='time',
            hop_length=chunk_size,
            backtrack=True,
            pre_max=20,
            post_max=20,
            pre_avg=50,
            post_avg=50,
            delta=0.1,
            wait=30
        )
        
        # Convert frames to milliseconds
        boundaries = [int(frame * 1000) for frame in onset_frames]
        
        # Add start and end points
        boundaries = [0] + boundaries + [len(self.audio)]
        
        return boundaries
        
    def segment_audio(self) -> List[Dict]:
        """
        Segment audio into chunks between min_segment_length and max_segment_length
        Returns list of segments with their information
        """
        boundaries = self.find_sentence_boundaries()
        current_segment_start = 0
        current_segment_length = 0
        segments_info = []
        
        for i, boundary in enumerate(boundaries[1:], 1):
            segment_length = boundary - current_segment_start
            
            # If adding this piece would make segment too long, cut at previous boundary
            if current_segment_length + segment_length > self.max_segment_length:
                # Save current segment if it's long enough
                if current_segment_length >= self.min_segment_length:
                    segment = self.audio[current_segment_start:boundaries[i-1]]
                    segments_info.append({
                        'start': current_segment_start / 1000,  # convert to seconds
                        'end': boundaries[i-1] / 1000,
                        'duration': current_segment_length / 1000,
                        'audio': segment
                    })
                
                # Start new segment
                current_segment_start = boundaries[i-1]
                current_segment_length = segment_length
            else:
                current_segment_length += segment_length
        
        # Add final segment if it's long enough
        if current_segment_length >= self.min_segment_length:
            segment = self.audio[current_segment_start:boundaries[-1]]
            segments_info.append({
                'start': current_segment_start / 1000,
                'end': boundaries[-1] / 1000,
                'duration': current_segment_length / 1000,
                'audio': segment
            })
            
        self.segments = segments_info
        return segments_info
    
    def export_segments(self, output_dir: str, base_filename: str) -> List[str]:
        """
        Export all segments to files and return list of file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        file_paths = []
        
        for i, segment in enumerate(self.segments, 1):
            # Generate filename with timestamp range
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            filename = f"{base_filename}_segment_{i:03d}_{start_time}-{end_time}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Export audio segment
            segment['audio'].export(filepath, format='wav')
            file_paths.append(filepath)
            
        return file_paths
    
    def create_zip_archive(self, file_paths: List[str], zip_path: str) -> str:
        """
        Create a zip archive containing all segment files
        Returns path to zip file
        """
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                zipf.write(file_path, os.path.basename(file_path))
        return zip_path

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM-SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}-{remaining_seconds:02d}" 