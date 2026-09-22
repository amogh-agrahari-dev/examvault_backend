import requests

ollama_url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma2:9b",
    "prompt": "Say hello in JSON format like {\"message\": \"hello\"}",
    "stream": False,
    "format": "json"
}
try:
    response = requests.post(ollama_url, json=payload, timeout=60)
    response.raise_for_status()
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
