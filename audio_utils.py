import os
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf
from typing import List, Tuple, Dict

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