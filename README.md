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

- Python 3.8+
- Ollama with llama3.3:latest model
- PyQt5 for the GUI
- FFmpeg (included in the repository)
- Various Python libraries (see requirements.txt)

## Installation

### Option 1: From GitHub

1. Clone this repository:
   ```
   git clone https://github.com/tsukisalami/LectureLogAI.git
   cd LectureLogAI
   ```

2. Create a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Ollama from [https://ollama.ai/](https://ollama.ai/)

4. Pull the llama3.3:latest model:
   ```
   ollama pull llama3.3:latest
   ```

5. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Run the application:
   ```
   python app/main.py
   ```

### Option 2: Local Installation

Follow the same steps as above, but instead of cloning from GitHub, you'll need to copy the project files to the target computer.

## Project Structure

```
LectureLogAI/
├── app/
│   ├── controllers/     # Application logic
│   ├── models/          # Data models
│   ├── views/           # UI components
│   ├── utils/           # Helper functions
│   └── data/            # Data storage
├── ffmpeg/              # FFmpeg binaries for audio processing
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

## Usage

1. Launch the application
2. Create a subject or select an existing one
3. Add a new class with date, name, and chapter
4. Record your class audio
5. Let the AI transcribe and summarize the content
6. View, edit, and organize your study materials

## Troubleshooting

- **FFmpeg issues**: If you encounter problems with FFmpeg, ensure the binaries are properly installed in the ffmpeg/ directory or install FFmpeg system-wide
- **Audio recording issues**: Check your microphone settings and permissions
- **Ollama connection errors**: Ensure Ollama is running and the llama3.3:latest model is installed

## License

[MIT License](LICENSE)
