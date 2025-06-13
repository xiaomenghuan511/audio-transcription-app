import os
import tempfile
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
from typing import List, Dict, Tuple, Optional
import asyncio
import concurrent.futures
import requests
import json
from datetime import datetime
import threading
import openai
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionModel:
    """Base class for transcription models"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._model = None
        self._lock = threading.Lock()
        
    def _load_model(self):
        """Load model if not already loaded"""
        pass
        
    async def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

class WhisperCloudModel(TranscriptionModel):
    """Transcription using OpenAI's Whisper Cloud API"""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized Whisper Cloud model")
    
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Whisper Cloud API."""
        try:
            logger.info(f"Starting transcription of {audio_path} with Whisper Cloud")
            with open(audio_path, "rb") as audio_file:
                logger.info(f"Calling OpenAI API for transcription...")
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                logger.info("Transcription completed successfully")
                return response.text
        except Exception as e:
            logger.error(f"Whisper Cloud API error: {str(e)}")
            raise

class WhisperLocalModel(TranscriptionModel):
    """Local Whisper model"""
    def __init__(self, model_size: str):
        super().__init__(f"whisper-local-{model_size}")
        self.model_size = model_size
        
    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = whisper.load_model(self.model_size)
        
    async def transcribe(self, audio_path: str) -> str:
        self._load_model()
        result = self._model.transcribe(audio_path)
        return result["text"]

class DeepgramModel(TranscriptionModel):
    """Deepgram API model"""
    def __init__(self, api_key: str):
        super().__init__("deepgram", api_key)
        
    async def transcribe(self, audio_path: str) -> str:
        """Use actual Deepgram API"""
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }
            
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(
                    "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                    headers=headers,
                    data=audio_file.read(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
                    return transcript
                else:
                    raise Exception(f"Deepgram API error: {response.status_code}")
                    
        except Exception as e:
            # Fallback to local model if API fails
            print(f"Deepgram API failed, falling back to local: {str(e)}")
            if self._model is None:
                with self._lock:
                    if self._model is None:
                        self._model = whisper.load_model("base")
            result = self._model.transcribe(audio_path)
            return result["text"]

class OptimizedSpeakerDiarizer:
    """Optimized Speaker diarization using pyannote.audio"""
    def __init__(self, hf_token: str):
        try:
            if not hf_token:
                raise ValueError("HuggingFace token is required")
                
            # First verify the token
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(
                "https://huggingface.co/pyannote/speaker-diarization",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid HuggingFace token")
            elif response.status_code == 403:
                raise ValueError("Please accept the user conditions at https://huggingface.co/pyannote/speaker-diarization")
            
            # Initialize the pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            )
            
            # Set device to CPU to avoid CUDA memory issues
            self.pipeline.to(torch.device("cpu"))
            self._cache = {}  # Cache diarization results
            
        except Exception as e:
            logger.error(f"Error initializing diarizer: {str(e)}")
            raise
        
    def diarize_full_audio(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on the full audio file once
        Returns list of segments with speaker labels and timestamps
        """
        try:
            if audio_path in self._cache:
                return self._cache[audio_path]
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process in smaller chunks if file is large
            audio = AudioSegment.from_file(audio_path)
            if len(audio) > 300000:  # If longer than 5 minutes
                logger.info("Long audio file detected, processing in chunks...")
                segments = self._process_large_audio(audio_path)
            else:
                diarization = self.pipeline(audio_path)
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'duration': turn.end - turn.start
                    })
            
            self._cache[audio_path] = segments
            return segments
            
        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            # Return empty segments instead of crashing
            return []
    
    def _process_large_audio(self, audio_path: str, chunk_duration: int = 300) -> List[Dict]:
        """Process large audio files in chunks"""
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000  # Convert to seconds
            segments = []
            
            for start_time in range(0, int(total_duration), chunk_duration):
                end_time = min(start_time + chunk_duration, total_duration)
                
                # Extract chunk
                chunk = audio[start_time * 1000:end_time * 1000]
                chunk_path = f"{audio_path}_chunk_{start_time}.wav"
                chunk.export(chunk_path, format="wav")
                
                # Process chunk
                diarization = self.pipeline(chunk_path)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        'start': turn.start + start_time,
                        'end': turn.end + start_time,
                        'speaker': speaker,
                        'duration': turn.end - turn.start
                    })
                
                # Clean up chunk file
                try:
                    os.remove(chunk_path)
                except:
                    pass
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return segments
            
        except Exception as e:
            logger.error(f"Error processing large audio: {str(e)}")
            return []
    
    def get_speakers_for_segment(self, full_diarization: List[Dict], 
                                segment_start: float, segment_end: float) -> List[Dict]:
        """
        Get speaker information for a specific segment from full diarization
        """
        segment_speakers = []
        for dia_segment in full_diarization:
            # Check if diarization segment overlaps with our audio segment
            overlap_start = max(dia_segment['start'], segment_start)
            overlap_end = min(dia_segment['end'], segment_end)
            
            if overlap_start < overlap_end:  # There is overlap
                segment_speakers.append({
                    'start': overlap_start - segment_start,  # Relative to segment
                    'end': overlap_end - segment_start,
                    'speaker': dia_segment['speaker'],
                    'duration': overlap_end - overlap_start
                })
        
        return segment_speakers

class OptimizedMultiModelTranscriber:
    """Optimized version with better performance"""
    def __init__(self, models: List[TranscriptionModel], diarizer: Optional[OptimizedSpeakerDiarizer] = None):
        self.models = models
        self.diarizer = diarizer
        self.speaker_mapping = {}
        logger.info(f"Initializing transcriber with {len(models)} models")
        
        # Pre-load all models
        for model in self.models:
            logger.info(f"Pre-loading model: {model.model_name}")
            model._load_model()
    
    def map_speaker_across_segments(self, speaker_id: str, segment_index: int) -> str:
        """Ensure consistent speaker naming across segments"""
        if speaker_id not in self.speaker_mapping:
            speaker_count = len(self.speaker_mapping)
            self.speaker_mapping[speaker_id] = f"Speaker {chr(65 + speaker_count)}"
        
        return self.speaker_mapping[speaker_id]
    
    async def transcribe_segment_batch(self, segment_info: Dict, full_diarization: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """Transcribe one segment with all models in parallel"""
        segment_path = segment_info['path']
        logger.info(f"Processing segment {segment_info['index']} from {segment_path}")
        
        # Create tasks for all models
        tasks = []
        for model in self.models:
            logger.info(f"Creating transcription task for {model.model_name}")
            task = asyncio.create_task(self._transcribe_single_model(model, segment_info, full_diarization))
            tasks.append((model.model_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for model_name, task in tasks:
            try:
                logger.info(f"Waiting for {model_name} transcription to complete")
                result = await task
                results[model_name] = result
                logger.info(f"{model_name} transcription completed successfully")
            except Exception as e:
                logger.error(f"Error in {model_name} transcription: {str(e)}")
                results[model_name] = {
                    'model': model_name,
                    'segment_index': segment_info['index'],
                    'error': str(e),
                    'transcript': f"[Error: {str(e)}]",
                    'formatted_output': f"[Error: {str(e)}]\n"
                }
        
        return results
    
    async def _transcribe_single_model(self, model: TranscriptionModel, 
                                     segment_info: Dict, 
                                     full_diarization: Optional[List[Dict]] = None) -> Dict:
        """Transcribe with a single model"""
        try:
            logger.info(f"Starting transcription with {model.model_name}")
            # Transcribe the segment
            transcript = await model.transcribe(segment_info['path'])
            logger.info(f"Transcription completed for {model.model_name}")
            
            # Get speaker information if diarization is enabled
            segment_speakers = []
            if self.diarizer and full_diarization:
                logger.info("Getting speaker information for segment")
                segment_speakers = self.diarizer.get_speakers_for_segment(
                    full_diarization, 
                    segment_info['start'], 
                    segment_info['end']
                )
            
            # Format output
            formatted_output = self.format_segment_output(
                transcript, segment_speakers, segment_info['start']
            )
            logger.info(f"Output formatted for {model.model_name}")
            
            return {
                'model': model.model_name,
                'segment_index': segment_info['index'],
                'start_time': segment_info['start'],
                'end_time': segment_info['end'],
                'transcript': transcript,
                'diarization': segment_speakers,
                'formatted_output': formatted_output
            }
            
        except Exception as e:
            logger.error(f"Error in transcription with {model.model_name}: {str(e)}")
            raise
    
    def format_segment_output(self, transcript: str, diarization: List[Dict], segment_start: float) -> str:
        """Format the output with timestamps and speaker labels"""
        timestamp = self.format_timestamp(segment_start)
        
        if not diarization:
            return f"[{timestamp}]\n{transcript}\n"
        
        # Use the dominant speaker for the segment
        dominant_speaker = max(diarization, key=lambda x: x['duration'])
        speaker = self.map_speaker_across_segments(dominant_speaker['speaker'], 0)
        
        return f"[{timestamp}]\n[{speaker}]: {transcript}\n"
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    async def transcribe_all_segments_optimized(self, segments_info: List[Dict], 
                                              original_audio_path: str) -> Dict[str, List[Dict]]:
        """Optimized transcription of all segments"""
        # Step 1: Perform diarization if enabled
        full_diarization = None
        if self.diarizer:
            print("Performing speaker diarization on full audio...")
            full_diarization = self.diarizer.diarize_full_audio(original_audio_path)
        
        # Step 2: Process segments in batches
        print(f"Transcribing {len(segments_info)} segments with {len(self.models)} models...")
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent segments
        
        async def process_segment_with_semaphore(segment_info):
            async with semaphore:
                return await self.transcribe_segment_batch(segment_info, full_diarization)
        
        # Process all segments
        segment_tasks = [process_segment_with_semaphore(seg) for seg in segments_info]
        segment_results = await asyncio.gather(*segment_tasks, return_exceptions=True)
        
        # Organize results by model
        results = {model.model_name: [] for model in self.models}
        
        for i, segment_result in enumerate(segment_results):
            if isinstance(segment_result, Exception):
                # Handle exception
                for model in self.models:
                    results[model.model_name].append({
                        'model': model.model_name,
                        'segment_index': i,
                        'error': str(segment_result),
                        'transcript': f"[Error: {str(segment_result)}]"
                    })
            else:
                # Normal result
                for model_name, model_result in segment_result.items():
                    results[model_name].append(model_result)
        
        return results
    
    def combine_results(self, results: Dict[str, List[Dict]]) -> Dict[str, str]:
        """Combine all segment results into final transcripts for each model"""
        final_transcripts = {}
        
        for model_name, model_results in results.items():
            # Sort by segment index
            model_results.sort(key=lambda x: x.get('segment_index', 0))
            
            # Combine all formatted outputs
            combined_text = ""
            for result in model_results:
                if 'formatted_output' in result:
                    combined_text += result['formatted_output'] + "\n"
                elif 'transcript' in result:
                    # Fallback formatting
                    timestamp = self.format_timestamp(result.get('start_time', 0))
                    combined_text += f"[{timestamp}]\n[Speaker A]: {result['transcript']}\n\n"
            
            final_transcripts[model_name] = combined_text.strip()
        
        return final_transcripts

# Legacy AudioAnalyzer class for backward compatibility
class AudioAnalyzer:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        
    def process_chunk(self, chunk_info, temp_dir):
        return {"transcript": "Legacy processing"}
    
    def format_output(self, results):
        return "Legacy output"
    
    def post_process(self, text):
        return text 