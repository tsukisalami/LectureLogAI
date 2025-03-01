@echo off
echo LectureLogAI Installation Script
echo ==============================
echo.

:: Check for Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.8 or higher.
    echo Visit https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Run the setup script
echo Running setup script...
python ..\..\setup\setup.py
if %errorlevel% neq 0 (
    echo Setup failed. Please try manual installation.
    pause
    exit /b 1
)

echo.
echo Installation completed!
echo.
echo To start LectureLogAI:
echo 1. Run "start.bat"
echo.

:: Create a start.bat file for easy launching
echo @echo off > ..\..\start.bat
echo call venv\Scripts\activate >> ..\..\start.bat
echo python app\main.py >> ..\..\start.bat
echo pause >> ..\..\start.bat

echo Created start.bat for easy launching.
echo.
pause 