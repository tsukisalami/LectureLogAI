import sys
print(f"Python version: {sys.version}")

modules_to_test = [
    "numpy", 
    "PyQt5", 
    "pyaudio", 
    "whisper", 
    "requests", 
    "pydub", 
    "matplotlib", 
    "PIL", 
    "sounddevice", 
    "wave", 
    "dotenv", 
    "ollama"
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module} imported successfully")
    except ImportError as e:
        print(f"✗ {module} could not be imported: {e}") 