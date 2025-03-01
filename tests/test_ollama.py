import requests
import json
import time
import sys

def test_ollama_connection(host="http://localhost:11434", model="mistral:latest"):
    """Test basic connectivity to Ollama server."""
    print(f"Testing Ollama connection to {host}...")
    
    # Test 1: Check if Ollama server is running
    try:
        print("\n--- Test 1: Basic connectivity ---")
        start_time = time.time()
        response = requests.get(f"{host}/api/tags", timeout=5)
        elapsed = time.time() - start_time
        
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Connection successful!")
            models = response.json().get("models", [])
            print(f"Available models: {[m.get('name') for m in models]}")
            
            # Check if our target model exists
            if any(m.get('name') == model for m in models):
                print(f"✅ Model '{model}' is available")
            else:
                print(f"❌ Model '{model}' is NOT available")
                print(f"   Available models: {[m.get('name') for m in models]}")
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.Timeout:
        print("❌ Connection timed out - Ollama server may not be running")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Ollama server may not be running")
    except Exception as e:
        print(f"❌ Unexpected error checking connection: {str(e)}")
    
    # Test 2: Try a simple generation with the model
    if model:
        try:
            print("\n--- Test 2: Simple generation ---")
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "prompt": "Say hello in one short sentence.",
                "stream": False,
                "max_tokens": 100
            }
            
            print(f"Sending request to {host}/api/generate...")
            start_time = time.time()
            response = requests.post(
                f"{host}/api/generate",
                headers=headers,
                data=json.dumps(data),
                timeout=10
            )
            elapsed = time.time() - start_time
            
            print(f"Response time: {elapsed:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print("✅ Generation successful!")
                print(f"Response: {response_text}")
            else:
                print(f"❌ Generation failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.Timeout:
            print("❌ Generation timed out after 10 seconds")
        except Exception as e:
            print(f"❌ Unexpected error during generation: {str(e)}")

if __name__ == "__main__":
    # Use command line arguments if provided
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "mistral:latest"
    
    print(f"Testing Ollama connection with host={host}, model={model}")
    test_ollama_connection(host, model)
    print("\nTest complete!") 