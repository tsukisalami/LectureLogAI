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
    
    def __init__(self, ollama_host="http://localhost:11434", model_name="mistral:latest", whisper_model_size="medium", summarization_preset="Reserved & concise"):
        """Initialize the AI processor with Ollama settings."""
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.whisper_model = None  # Lazy-load the model
        self.whisper_model_size = whisper_model_size
        self.summarization_preset = summarization_preset
        self.progress_callback = None
        self.safe_mode = False  # Safe mode flag for transcription
    
    def _load_whisper_model(self, model_size=None):
        """Load the Whisper model."""
        try:
            # Use provided size or instance default
            size = model_size or self.whisper_model_size
            
            # Log the load attempt
            logging.info(f"Loading Whisper model with size: {size}")
            
            # Check CUDA availability
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logging.info("CUDA is available, using GPU")
                device = "cuda"
                
                # Log GPU information if available
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    logging.info(f"Using GPU: {gpu_name}")
                    
                    gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    gpu_allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
                    
                    logging.info(f"GPU total memory: {gpu_memory_mb:.2f} MB")
                    logging.info(f"GPU allocated memory: {gpu_allocated_mb:.2f} MB")
                    
                    # Check if we have enough memory for the selected model
                    required_memory = {
                        "tiny": 1024,      # ~1 GB
                        "base": 1536,      # ~1.5 GB
                        "small": 2560,     # ~2.5 GB
                        "medium": 4608,    # ~4.5 GB
                        "large": 9216      # ~9 GB
                    }
                    
                    # Warning if low memory, but don't auto-downgrade unless necessary
                    if gpu_memory_mb < required_memory.get(size, 2000):
                        logging.warning(f"GPU memory may be insufficient for {size} model. Consider using a smaller model.")
                except Exception as e:
                    logging.warning(f"Could not get detailed GPU info: {str(e)}")
            else:
                logging.info("CUDA is not available, using CPU")
                device = "cpu"
                # If using CPU, warn for larger models
                if size not in ["tiny", "base"] and not model_size:
                    logging.warning("CPU detected but using a large model. This may be slow.")
                    logging.warning("Consider setting whisper_model_size to 'base' or 'tiny' for faster processing.")
            
            # Load the model (may take time for larger models)
            model = whisper.load_model(size, device=device)
            logging.info(f"Successfully loaded model on device: {model.device}")
            return model
            
        except Exception as e:
            logging.error(f"Error loading Whisper model: {str(e)}", exc_info=True)
            
            # If failed and not already trying tiny+CPU, attempt fallback
            if size != "tiny" or device != "cpu":
                logging.warning("Attempting fallback to tiny model on CPU")
                try:
                    return whisper.load_model("tiny", device="cpu")
                except Exception as fallback_e:
                    logging.error(f"Fallback model loading also failed: {str(fallback_e)}")
            
            # Re-raise the original exception
            raise
    
    def set_progress_callback(self, callback):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def transcribe_audio(self, audio_file_path, retry_with_fallback=True):
        """Transcribe the provided audio file.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            retry_with_fallback: Whether to retry with fallback method if transcription fails
            
        Returns:
            Dictionary containing transcription text and detected language
        """
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
            
            # Check if model is loaded, if not, load it
            if self.whisper_model is None:
                whisper_model = self._load_whisper_model()
                self.whisper_model = whisper_model
            else:
                whisper_model = self.whisper_model
                
            # Log model information
            if hasattr(whisper_model, 'device'):
                logging.info(f"Using Whisper model: {self.whisper_model_size} on {whisper_model.device}")
            
            # Set transcription options
            options = {
                "language": None,  # Allow automatic language detection
                "task": "transcribe",
                "verbose": True,
                "word_timestamps": False,  # Word timestamps can be very slow
                "patience": 2,
                "beam_size": 5,
                "best_of": 5,
                "initial_prompt": "Bonjour, comment allez-vous? Je m'appelle Claude. J'espère que vous allez bien aujourd'hui. Veuillez transcrire avec la ponctuation et les majuscules appropriées.",
                "fp16": torch.cuda.is_available(),
                "without_timestamps": True,  # Don't include timestamps in output
                "hallucination_silence_threshold": None,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "condition_on_previous_text": True,
                # Add important punctuation options
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                # Force punctuation and capitalization for all languages
                "suppress_tokens": [],  # Don't suppress any tokens
            }
            
            # Perform transcription
            logging.info(f"Starting transcription with options: {options}")
            transcription_result = whisper_model.transcribe(audio_file_path, **options)
            
            # Extract transcript text
            transcript = transcription_result.get('text', '')
            
            # Log identified language
            detected_language = transcription_result.get('language')
            logging.info(f"Whisper detected language: {detected_language}")
            
            # Process the transcript
            transcript = transcript.strip()
            
            # Log end of transcription
            elapsed_time = time.time() - start_time
            logging.info(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            # Return the transcription text and detected language
            return {
                "text": transcript,
                "language": detected_language
            }
        except Exception as e:
            import traceback
            error_msg = f"Error in transcription: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            
            # Try to clear GPU memory if possible
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info("Cleared GPU cache")
            except:
                pass
                
            # Enable safe mode for future transcriptions if this failed
            self.safe_mode = True
            
            if retry_with_fallback:
                # Try the fallback method
                logging.info("Attempting fallback transcription method")
                return self._fallback_transcribe(audio_file_path, custom_error=str(e))
            else:
                # Re-raise the exception
                raise
        finally:
            # Cleanup to help prevent memory leaks
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info("Cleared GPU cache")
            except:
                pass
    
    def _fallback_transcribe(self, audio_file_path, custom_error=None):
        """Fallback method for transcription when the primary method fails."""
        try:
            logging.info("Using fallback transcription method")
            
            # First try using the existing model with simpler options
            if self.whisper_model is not None:
                try:
                    logging.info("Attempting fallback with existing model using simplified options")
                    # Use very minimal options to avoid errors
                    simple_options = {
                        "language": None,  # Auto-detect language
                        "fp16": torch.cuda.is_available(),  # Use GPU if available
                        "task": "transcribe"
                    }
                    result = self.whisper_model.transcribe(audio_file_path, **simple_options)
                    transcript = result.get("text", "").strip()
                    detected_language = result.get("language", "unknown")
                    
                    logging.info(f"Fallback transcription completed with existing model. Language: {detected_language}")
                    return {
                        "text": transcript,
                        "language": detected_language
                    }
                except Exception as e:
                    logging.warning(f"Fallback with existing model failed: {str(e)}")
                    # Continue to next fallback approach
            
            # If that fails, load the tiny model on CPU as last resort
            try:
                model = whisper.load_model("tiny", device="cpu")
                logging.info("Loaded tiny model on CPU for fallback transcription")
            except Exception as e:
                logging.error(f"Failed to load tiny model: {str(e)}")
                return {
                    "text": f"Transcription failed: {custom_error or 'All transcription methods failed'}",
                    "language": "unknown"
                }
            
            # Use minimal options for stability
            options = {
                "language": None,  # Auto-detect language
                "fp16": False,     # Disable fp16 for CPU
                "task": "transcribe"
            }
            
            # Perform transcription
            result = model.transcribe(audio_file_path, **options)
            transcript = result.get("text", "").strip()
            detected_language = result.get("language", "unknown")
            
            logging.info(f"Fallback transcription completed with tiny CPU model. Language: {detected_language}")
            return {
                "text": transcript,
                "language": detected_language
            }
        except Exception as e:
            logging.error(f"Fallback transcription error: {str(e)}", exc_info=True)
            return {
                "text": f"Transcription failed: {str(e)}",
                "language": "unknown"
            }
    
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
    
    def summarize_text(self, text, num_sentences=10, language=None):
        """Summarize the provided text using LLM.
        
        Args:
            text: The text to summarize
            num_sentences: Target number of sentences in the summary
            language: The language code of the text (e.g., 'fr', 'en'), or None for auto-detection
            
        Returns:
            A string containing the summary
        """
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize"
        
        # Detect language if not provided
        if language:
            detected_language = language
            logging.info(f"Using provided language: {language}")
        else:
            detected_language = self._detect_language(text)
            logging.info(f"Auto-detected language: {detected_language}")
        
        # Determine settings based on preset
        temperature = 0.7  # Default
        max_length = 2000  # Default
        
        # Configure based on preset
        preset = self.summarization_preset
        system_prompt = ""
        user_prompt = ""
        
        # Define format instructions based on language
        format_instructions_en = (
            "Format your summary with the following structure:\n"
            "1. Start with a bold main title (use ** for bold)\n"
            "2. Create main sections with ## (second-level headings)\n"
            "3. Use ### for subsections if needed\n"
            "4. Use bullet points (- ) for individual items\n"
            "5. Use indented bullet points for examples under concepts\n"
            "6. Use **bold** for key terms and *italics* for emphasis\n"
            "7. Keep the summary well-structured and visually organized\n\n"
            "You have freedom to reorganize concepts in a logical way rather than strictly chronological, "
            "if it helps comprehension and readability."
        )
        
        format_instructions_fr = (
            "Formatez votre résumé avec la structure suivante:\n"
            "1. Commencez par un titre principal en gras (utilisez ** pour le gras)\n"
            "2. Créez des sections principales avec ## (titres de deuxième niveau)\n"
            "3. Utilisez ### pour les sous-sections si nécessaire\n"
            "4. Utilisez des puces (- ) pour les éléments individuels\n"
            "5. Utilisez des puces indentées pour les exemples sous les concepts\n"
            "6. Utilisez **gras** pour les termes clés et *italique* pour l'emphase\n"
            "7. Gardez le résumé bien structuré et visuellement organisé\n\n"
            "Vous avez la liberté de réorganiser les concepts de manière logique plutôt que strictement chronologique, "
            "si cela aide à la compréhension et à la lisibilité."
        )
        
        # Select appropriate format instructions
        format_instructions = format_instructions_fr if detected_language == "fr" else format_instructions_en
        
        if preset == "Reserved & concise":
            temperature = 0.3
            max_length = 1500
            if detected_language == "fr":
                system_prompt = (
                    "Vous êtes un assistant éducatif qui crée des résumés concis et précis de cours. "
                    "Concentrez-vous uniquement sur les informations explicitement mentionnées dans la transcription. "
                    "N'ajoutez pas d'informations, d'exemples ou de contextes qui n'ont pas été directement énoncés. "
                    "Créez un résumé clair mais bref en utilisant des puces. "
                    "Votre réponse DOIT être en français."
                )
                user_prompt = (
                    f"Créez un résumé concis de cette transcription de cours. "
                    f"Incluez uniquement les informations qui ont été explicitement mentionnées. "
                    f"N'ajoutez aucune information qui n'a pas été directement énoncée dans la transcription.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"IMPORTANT: Votre résumé DOIT être en français, dans la même langue que la transcription."
                )
            else:
                system_prompt = (
                    "You are an educational assistant that creates concise, accurate summaries of class lectures. "
                    "Focus only on information explicitly mentioned in the transcript. "
                    "Do not add information, examples, or context beyond what was directly stated. "
                    "Create a clear but brief summary using bullet points. "
                    "Your response MUST be in English or the same language as the transcript."
                )
                user_prompt = (
                    f"Create a concise summary of this class transcript. "
                    f"Include only information that was explicitly mentioned. "
                    f"Do not add any information that wasn't directly stated in the transcript.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"IMPORTANT: Your summary MUST be in the same language as the transcript (detected: {detected_language})."
                )
            
        elif preset == "Reserved & developed":
            temperature = 0.4
            max_length = 3000
            if detected_language == "fr":
                system_prompt = (
                    "Vous êtes un assistant éducatif qui crée des résumés détaillés et complets des cours. "
                    "Concentrez-vous uniquement sur les informations explicitement mentionnées dans la transcription. "
                    "Organisez les informations clairement avec des titres et des puces détaillées, mais n'ajoutez pas "
                    "d'informations ou d'exemples au-delà de ce qui a été énoncé dans la transcription. "
                    "Votre réponse DOIT être en français."
                )
                user_prompt = (
                    f"Créez un résumé détaillé de cette transcription de cours. "
                    f"Visez 10-15 puces avec des informations détaillées. "
                    f"Concentrez-vous uniquement sur les informations explicitement mentionnées dans la transcription. "
                    f"N'ajoutez aucune nouvelle information, exemples ou contexte au-delà de ce qui a été directement énoncé.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPTION:\n{text}\n\n"
                    f"IMPORTANT: Votre résumé DOIT être en français, dans la même langue que la transcription."
                )
            else:
                system_prompt = (
                    "You are an educational assistant that creates detailed, comprehensive summaries of class lectures. "
                    "Focus only on information explicitly mentioned in the transcript. "
                    "Organize information clearly with headings and detailed bullet points, but do not add information "
                    "or examples beyond what was stated in the transcript. "
                    "Your response MUST be in English or the same language as the transcript."
                )
                user_prompt = (
                    f"Create a detailed summary of this class transcript. "
                    f"Aim for 10-15 bullet points with detailed information. "
                    f"Focus only on information explicitly mentioned in the transcript. "
                    f"Do not add any new information, examples, or context beyond what was directly stated.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"IMPORTANT: Your summary MUST be in the same language as the transcript (detected: {detected_language})."
                )
            
        elif preset == "Outspoken & concise":
            temperature = 0.7
            max_length = 1500
            if detected_language == "fr":
                system_prompt = (
                    "Vous êtes un assistant éducatif expert qui crée des résumés concis et perspicaces des cours. "
                    "N'hésitez pas à compléter le contenu de la transcription avec des informations contextuelles "
                    "pertinentes pour améliorer la compréhension des concepts clés. "
                    "Créez un résumé clair et concis avec les points clés. "
                    "Votre réponse DOIT être en français."
                )
                user_prompt = (
                    f"Créez un résumé concis mais perspicace de cette transcription de cours. "
                    f"Tout en vous concentrant principalement sur le contenu de la transcription, vous pouvez ajouter un contexte pertinent "
                    f"ou clarifier des concepts pour faciliter la compréhension. "
                    f"Gardez le résumé concis mais assurez-vous qu'il soit précieux et perspicace.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPTION:\n{text}\n\n"
                    f"IMPORTANT: Votre résumé DOIT être en français, dans la même langue que la transcription."
                )
            else:
                system_prompt = (
                    "You are an expert educational assistant that creates concise, insightful summaries of class lectures. "
                    "Feel free to supplement the transcript content with relevant background information and context "
                    "to enhance understanding of key concepts. "
                    "Create a clear, concise summary with key points. "
                    "Your response MUST be in English or the same language as the transcript."
                )
                user_prompt = (
                    f"Create a concise but insightful summary of this class transcript. "
                    f"While focusing primarily on the transcript content, you may add relevant context "
                    f"or clarify concepts that would help understanding. "
                    f"Keep the summary concise but make sure it's valuable and insightful.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"IMPORTANT: Your summary MUST be in the same language as the transcript (detected: {detected_language})."
                )
            
        else:  # Outspoken & developed
            temperature = 0.8
            max_length = 3000
            if detected_language == "fr":
                system_prompt = (
                    "Vous êtes un assistant éducatif expert qui crée des résumés complets et perspicaces des cours. "
                    "Fournissez des explications détaillées des concepts clés et n'hésitez pas à les compléter avec des "
                    "informations contextuelles pertinentes, des exemples et du contexte pour améliorer la compréhension. "
                    "Organisez le résumé avec des titres clairs et des points détaillés pour une clarté et une valeur "
                    "pédagogique maximales. "
                    "Votre réponse DOIT être en français."
                )
                user_prompt = (
                    f"Créez un résumé complet et perspicace de cette transcription de cours. "
                    f"Utilisez des titres et des points détaillés pour organiser l'information clairement. "
                    f"Tout en vous concentrant sur le contenu de la transcription, vous devriez ajouter un contexte pertinent, "
                    f"des exemples et des explications qui amélioreraient la compréhension du matériel. "
                    f"Visez un résumé approfondi et éducatif.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPTION:\n{text}\n\n"
                    f"IMPORTANT: Votre résumé DOIT être en français, dans la même langue que la transcription."
                )
            else:
                system_prompt = (
                    "You are an expert educational assistant that creates comprehensive, insightful summaries of class lectures. "
                    "Provide detailed explanations of key concepts and feel free to supplement with relevant background "
                    "information, examples, and context to enhance understanding. "
                    "Organize the summary with clear headings and detailed points for maximum clarity and learning value. "
                    "Your response MUST be in English or the same language as the transcript."
                )
                user_prompt = (
                    f"Create a comprehensive and insightful summary of this class transcript. "
                    f"Use headings and detailed points to organize the information clearly. "
                    f"While focusing on the transcript content, you should add relevant context, examples, "
                    f"and explanations that would enhance understanding of the material. "
                    f"Aim for a thorough and educational summary.\n\n"
                    f"{format_instructions}\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"IMPORTANT: Your summary MUST be in the same language as the transcript (detected: {detected_language})."
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

    def _detect_language(self, text):
        """Detect the language of the text.
        
        Uses langdetect library if available, otherwise falls back to simple heuristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            str: Language code ('fr' for French, 'en' for English or anything else)
        """
        if not text:
            logging.warning("Empty text provided for language detection")
            return "en"  # Default to English for empty text
            
        try:
            # Get a sample of the text to detect language (first 1000 chars)
            # This is more efficient than analyzing the entire text
            sample = text[:1000]
            
            # Try to use langdetect if available
            try:
                from langdetect import detect, LangDetectException
                try:
                    lang = detect(sample)
                    logging.info(f"Language detected by langdetect: {lang}")
                    
                    # Map language codes to our supported languages
                    if lang == 'fr':
                        return 'fr'
                    else:
                        # For now we only have special handling for French, default other languages to English
                        return 'en' if lang == 'en' else 'en'
                except LangDetectException as e:
                    logging.warning(f"LangDetect exception: {str(e)}")
                    # Continue to fallback method
            except ImportError:
                logging.warning("langdetect not installed, using basic detection")
                
            # Basic detection - if langdetect is not available or fails
            french_indicators = [
                " et ", " le ", " la ", " les ", " un ", " une ", " du ", " des ", " ce ", " cette ",
                " que ", " qui ", " où ", " pour ", " avec ", " dans ", " votre ", " notre ", 
                " est ", " sont ", " être ", " avoir ", " mais ", " ou ", " donc ", " ainsi ", " car ",
                "bonjour", "merci", "au revoir", "s'il vous plaît", "monsieur", "madame",
                " français", " je suis ", " tu es ", " il est ", " elle est ", " nous sommes ",
                " vous êtes ", " ils sont ", " elles sont "
            ]
            
            # Normalize text for comparison
            normalized_text = " " + sample.lower() + " "
            
            # Count French indicators
            french_count = sum(1 for word in french_indicators if word in normalized_text)
            
            # If many French indicators found, likely French
            if french_count >= 3:
                logging.info(f"Detected as French (matched {french_count} indicators)")
                return "fr"
            else:
                logging.info(f"Detected as English (matched only {french_count} French indicators)")
                return "en"
                
        except Exception as e:
            logging.error(f"Error detecting language: {e}", exc_info=True)
            return "en"  # Default to English on error 

    def reset_whisper_model(self):
        """Reset the Whisper model, clearing it from memory."""
        try:
            # Clear references to the model
            self.whisper_model = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Reset Whisper model and cleared GPU cache")
            else:
                logging.info("Reset Whisper model")
                
            # Force a garbage collection
            import gc
            gc.collect()
            
            return True
        except Exception as e:
            logging.error(f"Error resetting Whisper model: {str(e)}")
            return False 