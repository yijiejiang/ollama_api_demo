#!/usr/bin/env python3
"""
Ollama API Client Example
This script demonstrates how to use the Ollama API to interact with locally running models.
"""

import requests
import json
import time
import sys
import re

# Base URL for Ollama API
OLLAMA_API_URL = "http://localhost:11434/api"

def list_models():
    """List all available models in Ollama."""
    response = requests.get(f"{OLLAMA_API_URL}/tags")
    if response.status_code == 200:
        return response.json().get("models", [])
    else:
        print(f"Error listing models: {response.status_code}")
        return []

def generate_text(model_name, prompt, system_prompt=None, temperature=0.7, max_tokens=500):
    """Generate text using the specified model."""
    # 使用流式处理，但在内部收集完整响应
    # 这样可以避免 JSON 解析问题
    url = f"{OLLAMA_API_URL}/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True  # 使用流式处理
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # 解码字节为字符串
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                        
                    json_response = json.loads(line)
                    chunk = json_response.get("response", "")
                    full_response += chunk
                    
                    # 检查是否完成
                    if json_response.get("done", False):
                        break
                except json.JSONDecodeError:
                    # 跳过无效的 JSON 行
                    continue
                except Exception as e:
                    print(f"Error processing generate response: {str(e)}")
                    continue
            
            return full_response
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def generate_text_stream(model_name, prompt, system_prompt=None, temperature=0.7):
    """Generate text using the specified model with streaming response."""
    url = f"{OLLAMA_API_URL}/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # Decode bytes to string if needed
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                        
                    json_response = json.loads(line)
                    chunk = json_response.get("response", "")
                    full_response += chunk
                    print(chunk, end="", flush=True)
                    sys.stdout.flush()  # Ensure output is flushed
                    
                    # Check if we're done
                    if json_response.get("done", False):
                        break
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
                except Exception as e:
                    print(f"\nError processing stream: {str(e)}")
                    continue
                    
            print()  # New line after completion
            return full_response
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def get_embeddings(model_name, text):
    """Get embeddings for the provided text using the specified model."""
    url = f"{OLLAMA_API_URL}/embeddings"
    
    payload = {
        "model": model_name,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            print(f"Error getting embeddings: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Exception getting embeddings: {str(e)}")
        return []

def chat_completion(model_name, messages, temperature=0.7):
    """Chat completion with message history."""
    # 使用流式处理，但在内部收集完整响应
    url = f"{OLLAMA_API_URL}/chat"
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": True  # 使用流式处理
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    # 解码字节为字符串
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                        
                    line_data = json.loads(line)
                    if "message" in line_data and "content" in line_data["message"]:
                        chunk = line_data["message"]["content"]
                        full_content += chunk
                except json.JSONDecodeError:
                    # 跳过无效的 JSON 行
                    continue
                except Exception as e:
                    print(f"Error processing chat response: {str(e)}")
                    continue
            
            # 返回一个模拟的响应对象
            return {
                "message": {
                    "role": "assistant",
                    "content": full_content
                }
            }
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def extract_json_objects(text):
    """Extract valid JSON objects from a string that might contain multiple JSON objects."""
    # Find all potential JSON objects in the text
    json_objects = []
    
    # Split the text by newlines to handle multiple JSON objects
    lines = text.split('\n')
    
    # Process each line as a potential JSON object
    for line in lines:
        if not line.strip():
            continue
            
        try:
            json_obj = json.loads(line)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Try to find valid JSON objects using regex
            try:
                # Look for patterns that might be valid JSON objects
                matches = re.findall(r'\{.*?\}', line)
                for match in matches:
                    try:
                        json_obj = json.loads(match)
                        json_objects.append(json_obj)
                    except:
                        pass
            except:
                pass
    
    return json_objects

if __name__ == "__main__":
    # List available models
    print("Available models:")
    models = list_models()
    for model in models:
        print(f"- {model.get('name')}: {model.get('size')}")
    print()
    
    # Choose a model from the available ones
    model_name = "qwen2.5:7b"  # You can change this to any available model
    
    # Example 1: Simple text generation
    print(f"\n=== Text Generation with {model_name} ===")
    prompt = "写一首关于春天的诗"
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    response = generate_text(model_name, prompt)
    end_time = time.time()
    
    print(f"Response: {response}")
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")
    
    # Example 2: Chat completion
    print(f"\n=== Chat Completion with {model_name} ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    start_time = time.time()
    response = chat_completion(model_name, messages)
    end_time = time.time()
    
    if isinstance(response, dict):
        if "error" in response:
            print(f"Error: {response['error']}")
        elif "message" in response and isinstance(response["message"], dict):
            print(f"Response: {response['message'].get('content', '')}")
        elif "response" in response:
            print(f"Response: {response['response']}")
        else:
            print(f"Unexpected response format: {response}")
    else:
        print(f"Unexpected response type: {type(response)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")
    
    # Example 3: Streaming text generation
    print(f"\n=== Streaming Text Generation with {model_name} ===")
    prompt = "Explain quantum computing in simple terms"
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    start_time = time.time()
    generate_text_stream(model_name, prompt)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")
    
    # Example 4: Get embeddings (using bge-m3 which is designed for embeddings)
    embedding_model = "bge-m3"
    print(f"\n=== Embeddings with {embedding_model} ===")
    text = "This is a sample text for embedding"
    
    start_time = time.time()
    embeddings = get_embeddings(embedding_model, text)
    end_time = time.time()
    
    if isinstance(embeddings, list):
        print(f"Embedding dimension: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
    else:
        print(f"Error getting embeddings: {embeddings}")
    
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")