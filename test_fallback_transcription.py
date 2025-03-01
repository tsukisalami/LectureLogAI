import os
import time
import requests
import sys
import json

def test_fallback_transcription(audio_path, ollama_host="http://localhost:11434", model_name="mistral:latest"):
    """Test the fallback transcription process that uses Ollama."""
    
    # Verify audio file exists
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        return False
    
    # Get basic audio file info
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"📊 Audio file: {os.path.basename(audio_path)} ({file_size_mb:.2f} MB)")
    
    # Test 1: Check Ollama connection and model availability
    print("\n--- Test 1: Checking Ollama availability ---")
    try:
        start_time = time.time()
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Ollama connection successful ({elapsed:.2f}s)")
            
            # Check for model
            available_models = response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            
            if model_name in model_names:
                print(f"✅ Model '{model_name}' is available")
            else:
                print(f"❌ Model '{model_name}' is NOT available!")
                print(f"   Available models: {model_names}")
                return False
                
        else:
            print(f"❌ Ollama connection failed: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Connection to Ollama timed out!")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Ollama - is it running?")
        return False
    except Exception as e:
        print(f"❌ Unexpected error checking Ollama: {e}")
        return False
    
    # Test 2: Try the fallback transcription message generation
    print("\n--- Test 2: Testing fallback transcription ---")
    try:
        # Create a simple prompt for Ollama (similar to the one in the app)
        system_prompt = (
            "You are an expert transcription assistant. Due to technical limitations, "
            "automated transcription is not available right now."
        )
        
        user_prompt = (
            f"I was trying to transcribe an audio file '{os.path.basename(audio_path)}' "
            f"but automated transcription is not working. Please provide a helpful message explaining "
            f"that I should try again later or use a different transcription service."
        )
        
        # Prepare request data
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "max_tokens": 2000
        }
        
        # Send the request with a timeout
        print(f"Requesting transcription message from Ollama ({model_name})...")
        print(f"Prompt length: {len(user_prompt)} characters")
        start_time = time.time()
        
        response = requests.post(
            f"{ollama_host}/api/generate",
            headers=headers,
            data=json.dumps(data),
            timeout=20  # Longer timeout for generation
        )
        
        elapsed = time.time() - start_time
        print(f"Response received in {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("response", "")
            
            if message:
                print("✅ Successfully received transcription message!")
                print("\nMessage preview (first 100 chars):")
                print(f"'{message[:100]}...'")
                return True
            else:
                print("❌ Received empty response from Ollama")
                return False
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 20 seconds!")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during request: {e}")
        return False

if __name__ == "__main__":
    # Get arguments from command line or use defaults
    if len(sys.argv) < 2:
        print("Usage: python test_fallback_transcription.py <path_to_audio_file> [ollama_host] [model_name]")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    ollama_host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:11434"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "mistral:latest"
    
    print(f"Testing fallback transcription with:")
    print(f"- Audio file: {audio_path}")
    print(f"- Ollama host: {ollama_host}")
    print(f"- Model name: {model_name}")
    
    success = test_fallback_transcription(audio_path, ollama_host, model_name)
    
    if success:
        print("\n✅ Fallback transcription test completed successfully!")
    else:
        print("\n❌ Fallback transcription test failed!") 