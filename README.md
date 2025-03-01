# LectureLogAI: AI-Powered Lecture Assistant

AI powered transcripts, summaries and flashcards from classes, lectures, meetings, and written notes!

## Overview

A Python application that helps you study more effectively by:
- Recording audio from your classes
- Transcribing the audio using AI (Ollama with llama3.3)
- Summarizing the content to help with studying
- Organizing your notes by subject, class, and chapter
- (Future feature) Creating flashcards from key concepts

## Features

- **Record**: Capture audio from your classes
- **Transcribe**: Convert audio to text using AI
- **Summarize**: Generate concise summaries of the transcribed content
- **Organize**: Structure your study materials by subject, class, and chapter
- **Flashcards**: (Coming soon) Create study flashcards from key concepts

## Requirements

- Python 3.8+ (3.10+ recommended)
- Ollama with llama3.3:latest model
- PyQt5 for the GUI
- FFmpeg for audio processing
- Microphone for recording
- Various Python libraries (see requirements.txt)

## Installation

### System Requirements

- **Windows 10/11**: Should work out of the box
- **macOS**: May need Xcode command-line tools and Homebrew
- **Linux**: Need appropriate audio and GUI libraries installed

### Quick Installation (Recommended)

1. **Windows Users**:
   - Double-click `setup/scripts/install.bat`
   - Follow the prompts
   - After installation, use `start.bat` to run the application

2. **macOS/Linux Users**:
   - Open Terminal in the project directory
   - Run: `chmod +x setup/scripts/install.sh && ./setup/scripts/install.sh`
   - After installation, use `./start.sh` to run the application

### Manual Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/tsukisalami/LectureLogAI.git
   cd LectureLogAI
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Ollama**:
   - Download from [https://ollama.ai/](https://ollama.ai/)
   - Install following the platform-specific instructions
   - Verify installation by running `ollama --version` in a terminal/command prompt

4. **Pull the llama3.3:latest model**:
   ```bash
   ollama pull llama3.3:latest
   ```
   This may take some time as it downloads several GB of model data.

5. **Install the required Python dependencies**:
   ```bash
   pip install -r setup/requirements.txt
   ```
   Note: On Windows, sometimes PyAudio installation fails. If it does, download and install a matching wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

6. **Set up FFmpeg**:
   - The application expects FFmpeg binaries in the `ffmpeg/` directory
   - If missing, download FFmpeg (static build) for your platform from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract the files and place the executables (ffmpeg, ffprobe) in the `ffmpeg/` directory

7. **Run the application**:
   ```bash
   python app/main.py
   ```

### Option 2: Local Installation (Without Git)

1. **Download the ZIP**:
   - Go to https://github.com/tsukisalami/LectureLogAI
   - Click the "Code" button and select "Download ZIP"
   - Extract the ZIP file to a location on your computer

2. Follow steps 2-7 from the GitHub installation method above.

## First Run Setup

When you first run the application:

1. The app will create required directories if they don't exist
2. You might be asked for microphone permissions
3. Create your first subject and class to get started
4. Test recording capability with a short sample

## Project Structure

```
LectureLogAI/
├── app/                 # Main application code
│   ├── controllers/     # Application logic
│   ├── models/          # Data models
│   ├── views/           # UI components
│   ├── utils/           # Helper functions
│   └── data/            # Data storage
├── setup/               # Installation and setup utilities
│   ├── scripts/         # Platform-specific installation scripts
│   └── requirements.txt # Python dependencies
├── tests/               # Test scripts and utilities
│   └── scripts/         # Test runner scripts
├── docs/                # Documentation (future use)
├── ffmpeg/              # FFmpeg binaries for audio processing
└── README.md            # Documentation
```

## Usage

1. Launch the application
2. Create a subject or select an existing one
3. Add a new class with date, name, and chapter
4. Record your class audio
5. Let the AI transcribe and summarize the content
6. View, edit, and organize your study materials

## Common Issues & Troubleshooting

### General Issues
- **Application won't start**: Ensure Python and all dependencies are installed correctly
- **UI looks strange**: Make sure PyQt5 is installed properly
- **Performance issues**: This app uses AI models that require significant RAM/CPU resources

### Recording Issues
- **No microphone detected**: Check your system's audio input settings
- **Permission denied**: Allow microphone access in your system settings
- **Recording silent**: Check that the correct microphone is selected and not muted

### FFmpeg Issues
- **FFmpeg not found**: Ensure FFmpeg binaries are in the ffmpeg/ directory
- **Conversion errors**: Make sure you have the latest version of FFmpeg
- **Missing codec**: Some formats may require additional codecs to be installed

### Ollama Issues
- **Connection refused**: Make sure Ollama is running in the background
- **Model not found**: Verify you've pulled the llama3.3:latest model
- **Slow responses**: The first run of AI functions may be slower as models load

### Platform-Specific Issues

#### Windows
- If installing PyAudio fails, install it from a wheel file: `pip install [path_to_wheel]/PyAudio‑0.2.11‑cp310‑cp310‑win_amd64.whl`
- Some antivirus software may interfere with recording

#### macOS
- You may need to grant explicit Terminal/app permissions for microphone access
- Install portaudio with Homebrew if PyAudio fails: `brew install portaudio`

#### Linux
- Install system packages: `sudo apt-get install python3-pyaudio python3-pyqt5 portaudio19-dev`
- Ensure PulseAudio or ALSA is properly configured

## License

[MIT License](LICENSE)
