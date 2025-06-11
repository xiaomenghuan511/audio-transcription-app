import os
import torch
import whisper
import numpy as np
from pyannote.audio import Pipeline
from speechbrain.pretrained import EncoderClassifier
import tensorflow as tf
import tensorflow_hub as hub
from typing import Dict, List, Tuple
import tempfile

class AudioAnalyzer:
    def __init__(self, hf_token: str):
        # Initialize models
        self.whisper_model = whisper.load_model("base")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
        self.emotion_model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-large"
        )
        # Load YAMNet for ambient sound classification
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper."""
        result = self.whisper_model.transcribe(audio_path)
        return result

    def perform_diarization(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization."""
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

    def detect_emotion(self, audio_path: str) -> str:
        """Detect emotion in speech segment."""
        waveform = self.emotion_model.load_audio(
            audio_path,
            savedir=tempfile.gettempdir()
        )
        batch = waveform.unsqueeze(0)
        predictions = self.emotion_model.classify_batch(batch)
        emotion_idx = predictions[0].argmax().item()
        return self.emotion_labels[emotion_idx]

    def classify_ambient_sound(self, audio_path: str) -> List[str]:
        """Classify ambient sounds using YAMNet."""
        waveform, sample_rate = tf.audio.decode_wav(
            tf.io.read_file(audio_path),
            desired_channels=1
        )
        waveform = tf.squeeze(waveform, axis=-1)
        
        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_classes = tf.argsort(class_scores, direction='DESCENDING')[:3]
        
        # Get class names from YAMNet
        class_names = self.yamnet_model.class_names
        ambient_sounds = [class_names[i].numpy().decode('utf-8') for i in top_classes]
        return ambient_sounds

    def process_chunk(self, chunk_info: Dict, temp_dir: str) -> Dict:
        """Process a single audio chunk."""
        # Save chunk temporarily
        temp_path = os.path.join(temp_dir, f"temp_chunk_{chunk_info['start_time']}.wav")
        chunk_info['audio_data'].export(temp_path, format='wav')
        
        result = {
            'start_time': chunk_info['start_time'],
            'duration': chunk_info['duration']
        }
        
        if chunk_info['has_speech']:
            # Transcribe
            transcription = self.transcribe_audio(temp_path)
            result['transcript'] = transcription['text']
            
            # Diarize
            speakers = self.perform_diarization(temp_path)
            result['speakers'] = speakers
            
            # Detect emotion
            emotion = self.detect_emotion(temp_path)
            result['emotion'] = emotion
        else:
            # Classify ambient sounds
            ambient_sounds = self.classify_ambient_sound(temp_path)
            result['ambient_sounds'] = ambient_sounds
        
        # Clean up
        os.remove(temp_path)
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
            timestamp = self.format_timestamp(result['start_time'])
            output_lines.append(f"\n[{timestamp}]")
            
            if 'transcript' in result:
                speaker_info = result['speakers'][0] if result['speakers'] else {'speaker': 'Unknown'}
                output_lines.append(
                    f"[{speaker_info['speaker']}], [{result['emotion']}]: {result['transcript']}"
                )
            else:
                sounds = ', '.join(result['ambient_sounds'])
                output_lines.append(f"[Ambient Sound - {sounds}]")
        
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