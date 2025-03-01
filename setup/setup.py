import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    """
    Setup script to help install LectureLogAI on a new computer.
    This handles creating the virtual environment and installing dependencies.
    """
    print("LectureLogAI Setup Script")
    print("=========================")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return 1
    
    print(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} detected.")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        print("\nCreating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return 1
    else:
        print("\nVirtual environment already exists.")

    # Determine activation script
    if platform.system() == "Windows":
        activate_script = os.path.join("venv", "Scripts", "activate")
        python_exe = os.path.join("venv", "Scripts", "python.exe")
        pip_exe = os.path.join("venv", "Scripts", "pip.exe")
    else:  # macOS or Linux
        activate_script = os.path.join("venv", "bin", "activate")
        python_exe = os.path.join("venv", "bin", "python")
        pip_exe = os.path.join("venv", "bin", "pip")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        if platform.system() == "Windows":
            subprocess.run([pip_exe, "install", "-r", "setup/requirements.txt"], check=True)
        else:
            # For Unix systems, we need to source the activate script, which requires shell=True
            subprocess.run(f"source {activate_script} && pip install -r setup/requirements.txt", 
                          shell=True, check=True, executable="/bin/bash")
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("You may need to manually install dependencies:")
        print("1. Activate the virtual environment")
        print("2. Run: pip install -r setup/requirements.txt")
        return 1

    # Check for FFmpeg
    ffmpeg_dir = Path("ffmpeg")
    if not ffmpeg_dir.exists():
        print("\nCreating FFmpeg directory...")
        os.makedirs("ffmpeg", exist_ok=True)
        
        print("\nNOTE: You need to download FFmpeg binaries and place them in the ffmpeg/ directory.")
        print("Download FFmpeg from: https://ffmpeg.org/download.html")
        if platform.system() == "Windows":
            print("Extract the ZIP file and copy ffmpeg.exe and ffprobe.exe to the ffmpeg/ directory.")
        elif platform.system() == "Darwin":  # macOS
            print("Or install with Homebrew: brew install ffmpeg")
            print("Then copy or symlink the binaries to the ffmpeg/ directory.")
        else:  # Linux
            print("Or install with your package manager: sudo apt install ffmpeg")
            print("Then copy or symlink the binaries to the ffmpeg/ directory.")
    
    # Remind about Ollama
    print("\nDon't forget to install Ollama:")
    print("1. Download from https://ollama.ai/")
    print("2. Install following the instructions for your platform")
    print("3. Pull the necessary model: ollama pull llama3.3:latest")
    
    print("\nSetup completed successfully!")
    print("To run LectureLogAI:")
    if platform.system() == "Windows":
        print("1. Activate the environment: venv\\Scripts\\activate")
    else:
        print("1. Activate the environment: source venv/bin/activate")
    print("2. Run the application: python app/main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 