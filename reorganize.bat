@echo off
echo Reorganizing project structure...

:: Create directories
mkdir setup 2>nul
mkdir setup\scripts 2>nul
mkdir tests 2>nul
mkdir docs 2>nul

:: Move setup files
move install.bat setup\scripts\ >nul 2>&1
move install.sh setup\scripts\ >nul 2>&1
move setup.py setup\ >nul 2>&1

:: Copy requirements.txt to setup directory (keeping the original)
copy requirements.txt setup\ >nul 2>&1

:: Move test files
move test_*.py tests\ >nul 2>&1
move run_tests.bat tests\scripts\ >nul 2>&1
move check_gpu.py tests\ >nul 2>&1

:: Create new README files
echo # Setup Instructions > setup\README.md
echo This directory contains installation scripts and setup utilities. >> setup\README.md
echo. >> setup\README.md
echo ## Contents >> setup\README.md
echo. >> setup\README.md
echo - `setup.py` - Main setup script >> setup\README.md
echo - `requirements.txt` - Python dependencies >> setup\README.md
echo - `scripts/` - Platform-specific installation scripts >> setup\README.md

echo # Tests > tests\README.md
echo This directory contains test scripts and utilities. >> tests\README.md
echo. >> tests\README.md
echo ## Running Tests >> tests\README.md
echo. >> tests\README.md
echo Use `scripts\run_tests.bat` to run all tests. >> tests\README.md

echo Project reorganization complete!
echo.
echo New structure:
echo - app/ - Main application code
echo - setup/ - Installation and setup files
echo - tests/ - Test scripts and utilities
echo - docs/ - Documentation
echo - ffmpeg/ - Audio processing binaries

:: Create a symbolic link for requirements.txt in root (for GitHub)
mklink requirements.txt setup\requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo Note: Created a copy of requirements.txt in setup/ directory
    echo       Original remains in the root directory
)

echo.
echo Note: You may need to update import paths in test files
echo       if they reference modules in the root directory. 