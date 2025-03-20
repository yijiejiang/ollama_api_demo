#!/usr/bin/env python3
"""
Simple Ollama Client
使用官方推荐的方式调用 Ollama API
"""

import json
import time
import requests
import sys

# Ollama API 基础 URL
OLLAMA_API_URL = "http://localhost:11434/api"

def list_models():
    """列出所有可用的模型"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return models
        else:
            print(f"获取模型列表失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        return []

def generate(model, prompt, system=None, stream=False, options=None):
    """使用指定模型生成文本"""
    url = f"{OLLAMA_API_URL}/generate"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    if system:
        payload["system"] = system
    if stream:
        payload["stream"] = stream
    if options:
        payload["options"] = options
    
    try:
        if stream:
            response = requests.post(url, json=payload, stream=True)
            if response.status_code != 200:
                print(f"生成文本失败: {response.status_code} - {response.text}")
                return None
            
            full_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    line_data = json.loads(line)
                    chunk = line_data.get("response", "")
                    full_text += chunk
                    print(chunk, end="", flush=True)
                    
                    if line_data.get("done", False):
                        break
                except json.JSONDecodeError:
                    print(f"\n解析响应时出错: {line}")
            
            print()  # 完成后换行
            return full_text
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"生成文本失败: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return None

def chat(model, messages, stream=False, options=None):
    """使用聊天模式与模型交互"""
    url = f"{OLLAMA_API_URL}/chat"
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    if stream:
        payload["stream"] = stream
    if options:
        payload["options"] = options
    
    try:
        if stream:
            response = requests.post(url, json=payload, stream=True)
            if response.status_code != 200:
                print(f"聊天失败: {response.status_code} - {response.text}")
                return None
            
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    line_data = json.loads(line)
                    if "message" in line_data:
                        chunk = line_data["message"].get("content", "")
                        full_content += chunk
                        print(chunk, end="", flush=True)
                except json.JSONDecodeError:
                    print(f"\n解析响应时出错: {line}")
            
            print()  # 完成后换行
            return full_content
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                if "message" in result:
                    return result["message"].get("content", "")
                return str(result)
            else:
                print(f"聊天失败: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"聊天时出错: {e}")
        return None

def embeddings(model, prompt):
    """获取文本的嵌入向量"""
    url = f"{OLLAMA_API_URL}/embeddings"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            print(f"获取嵌入向量失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return []

def interactive_chat(model, system_prompt=None):
    """交互式聊天界面"""
    print(f"\n=== 与 {model} 进行交互式聊天 ===")
    print("输入 'exit' 或 'quit' 结束聊天")
    print("输入 'clear' 清除对话历史")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.lower() == "clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("对话历史已清除")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\n助手: ", end="", flush=True)
        response = chat(model, messages, stream=True)
        
        if response:
            messages.append({"role": "assistant", "content": response})
        else:
            print("获取响应失败")

def main():
    # 列出可用模型
    print("可用模型:")
    models = list_models()
    for model in models:
        print(f"- {model.get('name')}: {model.get('size')}")
    print()
    
    # 选择默认模型
    default_model = "qwen2.5:7b"
    if not any(model.get('name').startswith(default_model) for model in models):
        if models:
            default_model = models[0].get('name')
        else:
            print("没有可用的模型")
            return
    
    # 示例 1: 生成文本
    print(f"\n=== 使用 {default_model} 生成文本 ===")
    prompt = "写一首关于春天的诗"
    print(f"提示词: {prompt}")
    
    start_time = time.time()
    response = generate(default_model, prompt)
    end_time = time.time()
    
    if response:
        print(f"响应: {response}")
    print(f"耗时: {end_time - start_time:.2f} 秒\n")
    
    # 示例 2: 聊天完成
    print(f"\n=== 使用 {default_model} 进行聊天 ===")
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下你自己"}
    ]
    
    start_time = time.time()
    response = chat(default_model, messages)
    end_time = time.time()
    
    if response:
        print(f"响应: {response}")
    print(f"耗时: {end_time - start_time:.2f} 秒\n")
    
    # 示例 3: 流式文本生成
    print(f"\n=== 使用 {default_model} 进行流式文本生成 ===")
    prompt = "解释一下量子计算的基本原理"
    print(f"提示词: {prompt}")
    print("响应: ", end="", flush=True)
    
    start_time = time.time()
    generate(default_model, prompt, stream=True)
    end_time = time.time()
    
    print(f"耗时: {end_time - start_time:.2f} 秒\n")
    
    # 示例 4: 获取嵌入向量
    embedding_model = "bge-m3"
    print(f"\n=== 使用 {embedding_model} 获取嵌入向量 ===")
    text = "这是一个用于嵌入的示例文本"
    
    start_time = time.time()
    embedding = embeddings(embedding_model, text)
    end_time = time.time()
    
    if embedding:
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"前 5 个值: {embedding[:5]}")
    print(f"耗时: {end_time - start_time:.2f} 秒\n")
    
    # 示例 5: 交互式聊天
    print("\n是否要启动交互式聊天? (y/n)")
    choice = input("> ")
    if choice.lower() in ["y", "yes"]:
        interactive_chat(default_model, "你是一个有帮助的助手。")

if __name__ == "__main__":
    main()