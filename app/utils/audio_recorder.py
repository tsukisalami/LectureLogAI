import pyaudio
import wave
import threading
import time
import os
import logging
from pathlib import Path

class AudioRecorder:
    """Utility for recording audio from the microphone."""
    
    def __init__(self, data_dir='app/data/recordings'):
        """Initialize the audio recorder with settings."""
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1  # Mono recording
        self.fs = 44100  # Sample rate (Hz)
        
        self.frames = []  # Stores recorded data
        self.recording = False
        self.paused = False
        self.audio = None
        self.stream = None
        self.data_dir = Path(data_dir)
        
        # Ensure the recordings directory exists
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            logging.info(f"Initialized AudioRecorder with data directory: {self.data_dir}")
        except Exception as e:
            logging.error(f"Failed to create recordings directory: {e}")
            print(f"Warning: Failed to create recordings directory: {e}")
        
        # Recording thread
        self.thread = None
    
    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            logging.warning("Attempted to start recording while already recording")
            return False
        
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Open stream
            self.stream = self.audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.fs,
                frames_per_buffer=self.chunk,
                input=True
            )
            
            self.recording = True
            self.paused = False
            self.frames = []
            
            # Start recording thread
            self.thread = threading.Thread(target=self._record)
            self.thread.daemon = True  # Ensure thread doesn't prevent app exit
            self.thread.start()
            
            logging.info("Audio recording started")
            return True
        except Exception as e:
            logging.error(f"Failed to start recording: {e}", exc_info=True)
            # Clean up any resources that might have been created
            self._cleanup_resources()
            return False
    
    def _record(self):
        """Record audio data in a separate thread."""
        try:
            while self.recording:
                if not self.paused:
                    try:
                        data = self.stream.read(self.chunk, exception_on_overflow=False)
                        self.frames.append(data)
                    except Exception as e:
                        logging.error(f"Error reading audio data: {e}", exc_info=True)
                        # Don't break the recording loop on temporary errors
                else:
                    time.sleep(0.1)
        except Exception as e:
            logging.error(f"Recording thread error: {e}", exc_info=True)
            self.recording = False
    
    def pause_recording(self):
        """Pause the recording."""
        if self.recording and not self.paused:
            self.paused = True
            logging.info("Recording paused")
            return True
        return False
    
    def resume_recording(self):
        """Resume a paused recording."""
        if self.recording and self.paused:
            self.paused = False
            logging.info("Recording resumed")
            return True
        return False
    
    def stop_recording(self):
        """Stop the recording and save the audio file."""
        if not self.recording:
            logging.warning("Attempted to stop recording when not recording")
            return None
        
        self.recording = False
        filepath = None
        
        try:
            # Wait for the recording thread to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)  # Wait up to 2 seconds
                if self.thread.is_alive():
                    logging.warning("Recording thread did not terminate in time")
            
            # Only save if we have recorded frames
            if not self.frames:
                logging.warning("No audio frames recorded")
                return None
                
            # Generate a filename based on the current time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = self.data_dir / filename
            
            # Save the recorded data as a WAV file
            wf = wave.open(str(filepath), 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.sample_format))
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            logging.info(f"Recording saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving recording: {e}", exc_info=True)
            filepath = None
        finally:
            # Always clean up resources
            self._cleanup_resources()
        
        return str(filepath) if filepath else None
    
    def _cleanup_resources(self):
        """Clean up audio resources."""
        try:
            # Stop and close the stream if it exists
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logging.error(f"Error closing audio stream: {e}")
                finally:
                    self.stream = None
            
            # Terminate PyAudio if it exists
            if hasattr(self, 'audio') and self.audio:
                try:
                    self.audio.terminate()
                except Exception as e:
                    logging.error(f"Error terminating PyAudio: {e}")
                finally:
                    self.audio = None
                    
            logging.info("Audio resources cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}", exc_info=True)
    
    def get_recording_length(self):
        """Get the current length of the recording in seconds."""
        return len(self.frames) * self.chunk / self.fs
    
    def is_recording(self):
        """Check if recording is in progress."""
        return self.recording
    
    def is_paused(self):
        """Check if recording is paused."""
        return self.paused
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.recording = False
        self._cleanup_resources() 