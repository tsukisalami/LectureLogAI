@echo off
echo Running Ollama connectivity tests...
echo ===============================

:: Test basic Ollama connectivity
echo Testing basic Ollama connectivity...
python ..\..\tests\test_ollama.py
echo.

:: Ask for audio file for the second test
echo Next, we'll test the fallback transcription with an audio file.
echo Please enter the path to an audio file:
set /p AUDIO_FILE="> "

if "%AUDIO_FILE%"=="" (
    echo No audio file provided, skipping fallback transcription test.
) else (
    echo Testing fallback transcription with %AUDIO_FILE%...
    python ..\..\tests\test_fallback_transcription.py "%AUDIO_FILE%"
)

echo.
echo Tests completed. Please check the results above.
pause 