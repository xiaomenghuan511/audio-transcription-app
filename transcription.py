import os
import whisper
import numpy as np
from typing import Dict, List, Tuple
import tempfile
import librosa
from pyannote.audio import Pipeline
import torch

class AudioAnalyzer:
    def __init__(self, hf_token: str):
        """Initialize the audio analyzer with necessary models."""
        try:
            self.whisper_model = whisper.load_model("base")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
            
        # Initialize diarization pipeline
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            )
        except Exception as e:
            print(f"Error initializing diarization pipeline: {str(e)}")
            print("\nPlease make sure you have:")
            print("1. A valid HuggingFace token")
            print("2. Accepted the user agreement at: https://huggingface.co/pyannote/speaker-diarization")
            print("3. Accepted the user agreement at: https://huggingface.co/pyannote/segmentation")
            raise Exception("Failed to initialize diarization pipeline")

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return {"text": "Transcription failed", "error": str(e)}

    def detect_speech_activity(self, audio_path: str) -> bool:
        """Simple speech activity detection using energy threshold."""
        try:
            y, sr = librosa.load(audio_path)
            energy = librosa.feature.rms(y=y)
            return np.mean(energy) > 0.01
        except Exception as e:
            print(f"Error in speech detection: {str(e)}")
            return True

    def perform_diarization(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization using pyannote.audio."""
        try:
            diarization = self.diarization_pipeline(audio_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                }
                speaker_segments.append(segment)
            
            return speaker_segments
        except Exception as e:
            print(f"Error in diarization: {str(e)}")
            return [{'start': 0, 'end': 0, 'speaker': 'Unknown'}]

    def process_chunk(self, chunk_info: Dict, temp_dir: str) -> Dict:
        """Process a single audio chunk."""
        # Save chunk temporarily
        temp_path = os.path.join(temp_dir, f"temp_chunk_{chunk_info['start_time']}.wav")
        chunk_info['audio_data'].export(temp_path, format='wav')
        
        result = {
            'start_time': chunk_info['start_time'],
            'duration': chunk_info['duration']
        }
        
        has_speech = self.detect_speech_activity(temp_path)
        
        if has_speech:
            # Perform diarization first
            speakers = self.perform_diarization(temp_path)
            result['speakers'] = []
            
            # For each speaker segment, extract audio and transcribe
            for speaker_segment in speakers:
                start_ms = int(speaker_segment['start'] * 1000)
                end_ms = int(speaker_segment['end'] * 1000)
                
                # Extract segment audio
                segment_audio = chunk_info['audio_data'][start_ms:end_ms]
                segment_path = os.path.join(temp_dir, f"temp_segment_{start_ms}.wav")
                segment_audio.export(segment_path, format='wav')
                
                # Transcribe segment
                transcription = self.transcribe_audio(segment_path)
                
                # Add to result
                result['speakers'].append({
                    'start': speaker_segment['start'],
                    'end': speaker_segment['end'],
                    'speaker': speaker_segment['speaker'],
                    'transcript': transcription['text']
                })
                
                # Clean up segment file
                try:
                    os.remove(segment_path)
                except:
                    pass
        else:
            result['ambient_sounds'] = ['background noise']
        
        # Clean up chunk file
        try:
            os.remove(temp_path)
        except:
            pass
        return result

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def format_output(self, results: List[Dict]) -> str:
        """Format processing results into readable output."""
        output_lines = []
        
        for result in results:
            if 'speakers' in result and result['speakers']:
                for speaker_info in result['speakers']:
                    # Calculate absolute timestamp
                    abs_start = result['start_time'] + speaker_info['start']
                    timestamp = self.format_timestamp(abs_start)
                    
                    if speaker_info['transcript'].strip():  # Only add non-empty transcripts
                        output_lines.append(
                            f"[{timestamp}] [{speaker_info['speaker']}]: {speaker_info['transcript']}"
                        )
            elif 'ambient_sounds' in result:
                timestamp = self.format_timestamp(result['start_time'])
                sounds = ', '.join(result['ambient_sounds'])
                output_lines.append(f"[{timestamp}] [Ambient Sound - {sounds}]")
        
        return '\n'.join(output_lines)

    def post_process(self, text: str) -> str:
        """Remove repetitive patterns and clean up the transcript."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines with excessive repetition
            words = line.split()
            if len(words) >= 5:
                repetition_count = sum(1 for i in range(len(words)-4)
                                    if words[i:i+5] == words[i]*5)
                if repetition_count > 0:
                    continue
            
            # Skip lines with mostly punctuation
            if len(line.strip()) > 0:
                punct_ratio = sum(1 for c in line if not c.isalnum()) / len(line)
                if punct_ratio < 0.5:
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) 