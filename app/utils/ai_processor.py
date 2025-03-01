import os
import json
import requests
import whisper
import subprocess
import tempfile
import logging
from pathlib import Path
import time
import torch

class AIProcessor:
    """Handles AI processing tasks like transcription and summarization."""
    
    def __init__(self, ollama_host="http://localhost:11434", model_name="mistral:latest", whisper_model_size="base", summarization_preset="Reserved & concise"):
        """Initialize the AI processor with Ollama settings."""
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.whisper_model = None  # Lazy-load the model
        self.whisper_model_size = whisper_model_size
        self.summarization_preset = summarization_preset
        self.progress_callback = None
        self.safe_mode = False  # Safe mode flag for transcription
    
    def _load_whisper_model(self, model_size=None):
        """Load the Whisper model for transcription.
        
        Args:
            model_size: Size of the model to load ("tiny", "base", "small", "medium", "large").
                        If None, uses the size from settings.
        """
        if self.whisper_model is None:
            # Use the provided model_size or fall back to the instance variable
            if model_size is None:
                model_size = self.whisper_model_size
            
            print(f"Loading Whisper model: {model_size}")
            logging.info(f"Loading Whisper model: {model_size}")
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            logging.info(f"Using device: {device}")
            
            # For GPU, check if we have enough VRAM for the selected model
            if device == "cuda":
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                required_memory = {
                    "tiny": 1024,      # ~1 GB
                    "base": 1536,      # ~1.5 GB
                    "small": 2560,     # ~2.5 GB
                    "medium": 4608,    # ~4.5 GB
                    "large": 9216      # ~9 GB
                }
                
                # Check if we have enough memory for the requested model
                if gpu_memory_mb < required_memory.get(model_size, 1536):
                    # If not enough memory, downgrade to a smaller model
                    if model_size == "large" and gpu_memory_mb >= required_memory["medium"]:
                        print(f"Insufficient GPU memory for 'large' model, downgrading to 'medium'")
                        model_size = "medium"
                    elif model_size in ["large", "medium"] and gpu_memory_mb >= required_memory["small"]:
                        print(f"Insufficient GPU memory for '{model_size}' model, downgrading to 'small'")
                        model_size = "small"
                    elif model_size in ["large", "medium", "small"] and gpu_memory_mb >= required_memory["base"]:
                        print(f"Insufficient GPU memory for '{model_size}' model, downgrading to 'base'")
                        model_size = "base"
                    elif gpu_memory_mb >= required_memory["tiny"]:
                        print(f"Limited GPU memory, using 'tiny' model")
                        model_size = "tiny"
                    else:
                        print(f"Very limited GPU memory, falling back to CPU")
                        device = "cpu"
            
            try:
                # Load the model with appropriate device
                self.whisper_model = whisper.load_model(model_size, device=device)
                print(f"Model loaded successfully on: {self.whisper_model.device}")
                logging.info(f"Model loaded successfully on: {self.whisper_model.device}")
            except Exception as e:
                print(f"Error loading model '{model_size}' on {device}: {e}")
                logging.error(f"Error loading model '{model_size}' on {device}: {e}", exc_info=True)
                
                # If failed and not already using tiny model on CPU, try fallback
                if model_size != "tiny" or device != "cpu":
                    print("Attempting fallback to tiny model on CPU...")
                    try:
                        self.whisper_model = whisper.load_model("tiny", device="cpu")
                        print(f"Fallback successful, model loaded on: {self.whisper_model.device}")
                        logging.info(f"Fallback successful, model loaded on: {self.whisper_model.device}")
                    except Exception as fallback_error:
                        print(f"Fallback failed: {fallback_error}")
                        logging.error(f"Fallback failed: {fallback_error}", exc_info=True)
                        raise
                else:
                    # No fallback possible
                    raise
        
        return self.whisper_model
    
    def set_progress_callback(self, callback):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe the audio file using Whisper."""
        try:
            # Check if audio file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
                
            # Verify audio file is valid
            try:
                import soundfile as sf
                info = sf.info(audio_file_path)
                print(f"Audio file info: {info.frames} frames, {info.samplerate} Hz, {info.channels} channels")
                logging.info(f"Audio file info: {info.frames} frames, {info.samplerate} Hz, {info.channels} channels")
            except Exception as e:
                print(f"Warning: Could not validate audio file: {e}")
                logging.warning(f"Could not validate audio file: {e}")
                # Continue anyway, as Whisper might still be able to handle it
            
            # Check if safe mode is enabled
            if self.safe_mode:
                return self._fallback_transcribe(audio_file_path)
            
            # Check for ffmpeg before proceeding with Whisper
            try:
                # Try to run a simple ffmpeg command to verify it's installed
                subprocess_result = subprocess.run(['ffmpeg', '-version'], 
                                                  stdout=subprocess.PIPE, 
                                                  stderr=subprocess.PIPE,
                                                  timeout=3)
                if subprocess_result.returncode != 0:
                    print("Warning: ffmpeg test command returned non-zero exit code")
                    logging.warning("ffmpeg test command returned non-zero exit code")
                    raise Exception("ffmpeg test failed")
            except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
                print(f"Error: ffmpeg is not installed or not in PATH: {str(e)}")
                logging.error(f"ffmpeg is not installed or not in PATH: {str(e)}")
                
                # Try to automatically install ffmpeg
                print("Attempting automatic ffmpeg installation...")
                install_success, install_message = self._attempt_ffmpeg_install()
                
                if install_success:
                    print(f"Auto-installation succeeded: {install_message}")
                    logging.info(f"Auto-installation succeeded: {install_message}")
                    # Continue with transcription since ffmpeg is now installed
                else:
                    print(f"Auto-installation failed: {install_message}")
                    logging.error(f"Auto-installation failed: {install_message}")
                    
                    # Create a detailed error message
                    ffmpeg_error_msg = (
                        "Transcription requires ffmpeg to be installed and available in your PATH.\n\n"
                        f"Automatic installation failed: {install_message}\n\n"
                        "Please install ffmpeg manually and try again, or use Safe Mode for transcription."
                    )
                    
                    # Enable safe mode and try fallback
                    self.safe_mode = True
                    print("Safe mode enabled due to missing ffmpeg")
                    logging.info("Safe mode enabled due to missing ffmpeg")
                    
                    # Try the fallback method with the specific error
                    return self._fallback_transcribe(audio_file_path, 
                                                   custom_error=ffmpeg_error_msg)
            
            # Load the model - add a memory check before loading
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                print(f"Available system memory: {available_memory_gb:.2f} GB")
                logging.info(f"Available system memory: {available_memory_gb:.2f} GB")
                
                # If less than 2GB of RAM available, warn that this might be problematic
                if available_memory_gb < 2.0:
                    print("Warning: Low system memory may affect transcription performance")
                    logging.warning("Low system memory may affect transcription performance")
            except ImportError:
                print("psutil not installed - skipping memory check")
                logging.warning("psutil not installed - skipping memory check")
            
            model = self._load_whisper_model()
            
            # Verify device
            device_info = f"Device: {model.device} "
            if torch.cuda.is_available():
                device_info += f"({torch.cuda.get_device_name(0)})"
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                # Check GPU memory
                try:
                    gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    print(f"GPU total memory: {gpu_memory_mb:.2f} MB")
                    
                    # Get current GPU memory usage
                    allocated_memory_mb = torch.cuda.memory_allocated(0) / (1024**2)
                    print(f"GPU allocated memory: {allocated_memory_mb:.2f} MB")
                    
                    # Check if we're close to GPU memory limits
                    if allocated_memory_mb > 0.8 * gpu_memory_mb:
                        print("Warning: High GPU memory usage - may affect performance")
                        logging.warning("High GPU memory usage - may affect performance")
                except Exception as e:
                    print(f"Could not check GPU memory: {e}")
                    logging.warning(f"Could not check GPU memory: {e}")
            else:
                device_info += "(CPU)"
                print("Using CPU for transcription")
                logging.info("Using CPU for transcription")
            
            print(device_info)
            
            # Check file size and audio duration
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            print(f"Audio file size: {file_size_mb:.2f} MB")
            
            # For very large files, provide a warning
            if file_size_mb > 100:
                print("Warning: Large audio file - transcription may take significant time and resources")
                logging.warning("Large audio file - transcription may take significant time and resources")
            
            # Log start of transcription
            print(f"Starting transcription of {audio_file_path}")
            logging.info(f"Starting transcription of {audio_file_path}")
            start_time = time.time()
            
            # Create transcription options with appropriate settings
            fp16 = torch.cuda.is_available()  # Use fp16 only on GPU
            options = {
                "fp16": fp16,
                "language": "en"  # You can adjust or make this configurable
            }
            
            # Transcribe the audio with progress callback if available
            try:
                # Set a timeout for the operation (not directly supported by Whisper,
                # but we can monitor the operation from another thread if needed)
                result = model.transcribe(audio_file_path, **options)
            except torch.cuda.OutOfMemoryError:
                # If we run out of GPU memory, try to recover by moving to CPU
                print("GPU out of memory, attempting to continue on CPU...")
                logging.warning("GPU out of memory, attempting to continue on CPU...")
                torch.cuda.empty_cache()  # Free up GPU memory
                
                # Recreate the model on CPU
                self.whisper_model = None  # Force reload
                self.whisper_model = whisper.load_model("base", device="cpu")
                print(f"Reloaded model on CPU: {self.whisper_model.device}")
                logging.info(f"Reloaded model on CPU: {self.whisper_model.device}")
                
                # Try again on CPU
                options["fp16"] = False  # Disable fp16 on CPU
                result = self.whisper_model.transcribe(audio_file_path, **options)
            
            # Log end of transcription
            elapsed_time = time.time() - start_time
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            logging.info(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            # Return the transcription text
            return result["text"]
        except Exception as e:
            import traceback
            error_msg = f"Transcription error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            logging.error(f"Transcription error: {str(e)}", exc_info=True)
            
            # When an error occurs, enable safe mode for future transcriptions
            self.safe_mode = True
            print("Safe mode enabled for future transcriptions")
            logging.info("Safe mode enabled for future transcriptions")
            
            # Try the fallback method instead
            try:
                print("Attempting fallback transcription method...")
                logging.info("Attempting fallback transcription method...")
                return self._fallback_transcribe(audio_file_path)
            except Exception as fallback_error:
                fallback_error_msg = f"Fallback transcription error: {str(fallback_error)}"
                print(fallback_error_msg)
                logging.error(f"Fallback transcription error: {str(fallback_error)}", exc_info=True)
                raise Exception(f"All transcription methods failed: {str(e)} and then {str(fallback_error)}")
        finally:
            # Cleanup to help prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared GPU cache")
    
    def _fallback_transcribe(self, audio_file_path, custom_error=None):
        """Fallback transcription method using simpler approach.
        
        This method is used when the Whisper model fails to load or transcribe.
        It should NEVER crash the application.
        """
        print("Using fallback transcription method...")
        logging.info("Using fallback transcription method...")
        
        # If we have a custom error (like missing ffmpeg), use it as the base message
        if custom_error:
            fallback_message = custom_error
        else:
            fallback_message = (
                "Automated transcription is currently unavailable. "
                "The system encountered an issue while processing your audio file. "
                "\n\nPlease try again later or use an external transcription service."
            )
        
        try:
            # First verify the audio file exists
            if not os.path.exists(audio_file_path):
                return f"{fallback_message}\n\nError: Audio file not found: {os.path.basename(audio_file_path)}"
            
            # Check for Ollama availability - fast path to avoid hanging
            try:
                # Use a very short timeout for quick check
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
                if response.status_code != 200:
                    return f"{fallback_message}\n\nError: Ollama server returned status code {response.status_code}.\n\nPlease ensure Ollama is running."
            except requests.exceptions.RequestException as e:
                # Any connection error - timeout, connection refused, etc.
                logging.error(f"Could not connect to Ollama service: {str(e)}")
                return f"{fallback_message}\n\nError: Could not connect to Ollama service at {self.ollama_host}.\n\nPlease ensure Ollama is installed and running."
            
            # Just use Ollama as a fallback
            print("Using Ollama for transcription message...")
            
            # First check if Ollama is available and has the requested model
            model_available, model_status = self.check_ollama_model()
            if not model_available:
                # If model not available, return clear error message
                error_msg = f"{fallback_message}\n\nModel Error: {model_status}\n\nPlease check the Settings menu to select an available model."
                logging.warning(f"Ollama model unavailable: {model_status}")
                return error_msg
            
            # Create a simple prompt for Ollama
            system_prompt = (
                "You are an expert transcription assistant. Due to technical limitations, "
                "automated transcription is not available right now."
            )
            
            user_prompt = (
                f"I was trying to transcribe an audio file '{os.path.basename(audio_file_path)}' "
                f"but automated transcription is not working. Please provide a helpful message explaining "
                f"that I should try again later or use a different transcription service."
            )
            
            # Generate a message using Ollama with timeout handling
            try:
                # Try to connect to Ollama with a timeout
                result = self._generate_with_ollama(user_prompt, system=system_prompt)
                
                if result and not result.startswith("Error:"):
                    # If successful, return the Ollama-generated message
                    return f"{fallback_message}\n\n{result}"
                else:
                    # If Ollama fails, log the error and return the default message
                    logging.warning(f"Ollama failed to generate message: {result}")
                    return f"{fallback_message}\n\nAdditional error: {result}\n\nPlease check that Ollama is running and a valid model is selected."
                    
            except Exception as ollama_error:
                # If Ollama fails entirely, log and return the default message
                logging.error(f"Could not connect to Ollama: {str(ollama_error)}", exc_info=True)
                return f"{fallback_message}\n\nAdditional info: Could not connect to Ollama service."
                
        except Exception as e:
            # Catch-all for any other errors
            logging.error(f"Fallback transcription failed: {str(e)}", exc_info=True)
            return f"{fallback_message}\n\nPlease check application logs for details."
    
    def _generate_with_ollama(self, prompt, system=None, max_tokens=2000, temperature=0.7):
        """Generate text using the Ollama API."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            data["system"] = system
        
        try:
            # Add a 10 second timeout to prevent hanging indefinitely
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                headers=headers,
                data=json.dumps(data),
                timeout=10  # Add a 10 second timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return f"Error: {response.status_code}"
        except requests.exceptions.Timeout:
            print("Timeout connecting to Ollama")
            return "Error: Timeout connecting to Ollama service. Please ensure Ollama is running."
        except Exception as e:
            print(f"Exception: {e}")
            return f"Error: {e}"
    
    def summarize_text(self, text, num_sentences=10):
        """Summarize the transcribed text based on the selected preset."""
        # Determine settings based on preset
        temperature = 0.7  # Default
        max_length = 2000  # Default
        
        # Configure based on preset
        preset = self.summarization_preset
        system_prompt = ""
        user_prompt = ""
        
        if preset == "Reserved & concise":
            temperature = 0.3
            max_length = 1500
            system_prompt = (
                "You are an educational assistant that creates concise, accurate summaries of class lectures. "
                "Focus only on information explicitly mentioned in the transcript. "
                "Do not add information, examples, or context beyond what was directly stated. "
                "Create a clear but brief summary using bullet points."
            )
            user_prompt = (
                f"Create a concise summary of this class transcript using 5-8 bullet points. "
                f"Include only information that was explicitly mentioned. "
                f"Do not add any information that wasn't directly stated in the transcript.\n\n"
                f"TRANSCRIPT:\n{text}"
            )
            
        elif preset == "Reserved & developed":
            temperature = 0.4
            max_length = 3000
            system_prompt = (
                "You are an educational assistant that creates detailed, comprehensive summaries of class lectures. "
                "Focus only on information explicitly mentioned in the transcript. "
                "Organize information clearly with headings and detailed bullet points, but do not add information "
                "or examples beyond what was stated in the transcript."
            )
            user_prompt = (
                f"Create a detailed summary of this class transcript using headings and bullet points. "
                f"Aim for 10-15 bullet points with detailed information. "
                f"Focus only on information explicitly mentioned in the transcript. "
                f"Do not add any new information, examples, or context beyond what was directly stated.\n\n"
                f"TRANSCRIPT:\n{text}"
            )
            
        elif preset == "Outspoken & concise":
            temperature = 0.7
            max_length = 1500
            system_prompt = (
                "You are an expert educational assistant that creates concise, insightful summaries of class lectures. "
                "Feel free to supplement the transcript content with relevant background information and context "
                "to enhance understanding of key concepts. "
                "Create a clear, concise summary with key points."
            )
            user_prompt = (
                f"Create a concise but insightful summary of this class transcript using 5-8 key points. "
                f"While focusing primarily on the transcript content, you may add relevant context "
                f"or clarify concepts that would help understanding. "
                f"Keep the summary concise but make sure it's valuable and insightful.\n\n"
                f"TRANSCRIPT:\n{text}"
            )
            
        else:  # Outspoken & developed
            temperature = 0.8
            max_length = 3000
            system_prompt = (
                "You are an expert educational assistant that creates comprehensive, insightful summaries of class lectures. "
                "Provide detailed explanations of key concepts and feel free to supplement with relevant background "
                "information, examples, and context to enhance understanding. "
                "Organize the summary with clear headings and detailed points for maximum clarity and learning value."
            )
            user_prompt = (
                f"Create a comprehensive and insightful summary of this class transcript. "
                f"Use headings and detailed points to organize the information clearly. "
                f"While focusing on the transcript content, you should add relevant context, examples, "
                f"and explanations that would enhance understanding of the material. "
                f"Aim for a thorough and educational summary.\n\n"
                f"TRANSCRIPT:\n{text}"
            )
        
        # Generate the summary using Ollama with our preset-specific parameters
        summary = self._generate_with_ollama(
            user_prompt, 
            system=system_prompt,
            max_tokens=max_length,
            temperature=temperature
        )
        
        return summary
    
    def generate_flashcards(self, text, num_cards=5):
        """Generate flashcards from the transcribed text."""
        # Create a system prompt for flashcard generation
        system_prompt = (
            "You are an expert at creating educational flashcards. "
            "Create clear, focused flashcards with concise questions and comprehensive answers. "
            "Each flashcard should test understanding of a single concept."
        )
        
        # Create the user prompt
        user_prompt = (
            f"Please create {num_cards} flashcards based on the following transcript of a class lecture. "
            f"Each flashcard should have a question on one side and an answer on the other. "
            f"Focus on key concepts, definitions, and important facts.\n\n"
            f"Format each flashcard as:\n"
            f"Q: [Question]\nA: [Answer]\n\n"
            f"TRANSCRIPT:\n{text}"
        )
        
        # Generate the flashcards using Ollama
        flashcards_text = self._generate_with_ollama(user_prompt, system=system_prompt)
        
        # Parse the flashcards into a list of dictionaries
        flashcards = []
        current_question = None
        current_answer = ""
        
        for line in flashcards_text.split('\n'):
            line = line.strip()
            
            if line.startswith('Q:'):
                # If we already have a question, save the previous flashcard
                if current_question:
                    flashcards.append({
                        'question': current_question,
                        'answer': current_answer.strip()
                    })
                
                # Start a new flashcard
                current_question = line[2:].strip()
                current_answer = ""
            elif line.startswith('A:') and current_question:
                current_answer = line[2:].strip()
            elif current_question and current_answer:
                # Append to the current answer
                current_answer += f"\n{line}"
        
        # Add the last flashcard if there is one
        if current_question:
            flashcards.append({
                'question': current_question,
                'answer': current_answer.strip()
            })
        
        return flashcards
        
    def check_ollama_model(self):
        """Check if the currently selected Ollama model is available.
        
        Returns:
            tuple: (is_available, message) where is_available is a boolean and
                   message is a string with status or error information
        """
        try:
            # Check if Ollama is running with a timeout
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama server is not running"
            
            # Check if the model is available
            available_models = response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            
            if self.model_name not in model_names:
                return False, f"Model '{self.model_name}' is not available. Please run 'ollama pull {self.model_name}' or select a different model."
            
            return True, f"Model '{self.model_name}' is available"
        except requests.exceptions.Timeout:
            return False, "Timeout connecting to Ollama server"
        except Exception as e:
            return False, f"Error connecting to Ollama: {e}"
    
    def _attempt_ffmpeg_install(self):
        """Attempt to download and install ffmpeg for the user.
        
        Returns:
            tuple: (success, message) where success is a boolean indicating if 
                   installation was successful, and message is a helpful string.
        """
        try:
            import platform
            
            # Only Windows is supported for automatic installation right now
            if platform.system() != "Windows":
                return False, "Automatic ffmpeg installation is only supported on Windows."
            
            # Create a temporary directory
            import tempfile
            import zipfile
            import urllib.request
            import shutil
            import os
            
            # Inform the user
            print("Attempting to download ffmpeg for Windows...")
            logging.info("Attempting to download ffmpeg for Windows...")
            
            # Create a temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download URL for ffmpeg (you may want to update this URL over time)
                ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
                zip_path = os.path.join(temp_dir, "ffmpeg.zip")
                
                # Download the ZIP file
                print(f"Downloading ffmpeg from {ffmpeg_url}...")
                urllib.request.urlretrieve(ffmpeg_url, zip_path)
                
                # Extract the ZIP file
                print("Extracting ffmpeg...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the ffmpeg.exe file
                ffmpeg_dir = None
                for root, dirs, files in os.walk(temp_dir):
                    if 'ffmpeg.exe' in files:
                        ffmpeg_dir = root
                        break
                
                if not ffmpeg_dir:
                    return False, "Could not find ffmpeg.exe in the downloaded package."
                
                # Create a directory for ffmpeg in the app's directory if it doesn't exist
                local_ffmpeg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ffmpeg")
                os.makedirs(local_ffmpeg_dir, exist_ok=True)
                
                # Copy the ffmpeg executable and DLLs to the ffmpeg directory
                ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                local_ffmpeg_exe = os.path.join(local_ffmpeg_dir, "ffmpeg.exe")
                shutil.copy2(ffmpeg_exe, local_ffmpeg_exe)
                
                # Add the local ffmpeg directory to the PATH environment variable for this process
                os.environ["PATH"] = local_ffmpeg_dir + os.pathsep + os.environ["PATH"]
                
                # Verify the installation
                try:
                    subprocess_result = subprocess.run(['ffmpeg', '-version'], 
                                                      stdout=subprocess.PIPE, 
                                                      stderr=subprocess.PIPE,
                                                      timeout=3)
                    if subprocess_result.returncode == 0:
                        return True, f"ffmpeg was downloaded and installed successfully to {local_ffmpeg_dir}"
                    else:
                        return False, f"ffmpeg was downloaded but verification failed with exit code {subprocess_result.returncode}"
                except Exception as verify_error:
                    return False, f"ffmpeg was downloaded but verification failed: {str(verify_error)}"
                
        except Exception as e:
            logging.error(f"Error attempting to install ffmpeg: {str(e)}", exc_info=True)
            return False, f"Could not download and install ffmpeg: {str(e)}"
    
    def get_available_ollama_models(self):
        """Get a list of available Ollama models using the 'ollama list' command."""
        try:
            # First try the API method
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                # Extract model names from the response
                models = []
                for model in data.get('models', []):
                    models.append(model.get('name'))
                if models:
                    logging.info(f"Found {len(models)} models via API")
                    return models
            
            # If API fails or returns no models, try the command line approach
            logging.info("Attempting to get models via command line")
            result = subprocess.run(['ollama', 'list'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  timeout=5)
            
            if result.returncode == 0:
                # Parse the output to extract model names
                lines = result.stdout.strip().split('\n')
                # Skip the header line if it exists
                if lines and "NAME" in lines[0]:
                    lines = lines[1:]
                
                models = []
                for line in lines:
                    if line.strip():
                        # The model name is typically the first part of the line
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                
                logging.info(f"Found {len(models)} models via command line")
                return models
            else:
                logging.error(f"Command failed with error: {result.stderr}")
                return []
                
        except Exception as e:
            logging.error(f"Error getting Ollama models: {str(e)}")
            return []
    
    def update_ollama_settings(self, host=None, model=None, whisper_model_size=None, summarization_preset=None):
        """Update Ollama settings."""
        if host:
            self.ollama_host = host
        if model:
            self.model_name = model
        if whisper_model_size:
            self.whisper_model_size = whisper_model_size
            # Force reload of Whisper model with new size
            self.whisper_model = None
        if summarization_preset:
            self.summarization_preset = summarization_preset
        return True 