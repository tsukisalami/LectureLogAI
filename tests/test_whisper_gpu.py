import torch
import whisper

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    
# Load a model
print("Loading Whisper model...")
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded on: {model.device}")

# Test with a simple transcription (optional)
# result = model.transcribe("path/to/your/audio.mp3")
# print(result["text"])

print("Whisper GPU test completed successfully!") 