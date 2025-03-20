#!/usr/bin/env python3
"""
Advanced Ollama Client
面向对象的 Ollama API 客户端，提供更多高级功能和错误处理
"""

import requests
import json
import time
import sys
import argparse
import os
from typing import Dict, List, Any, Optional, Union, Tuple

class OllamaClient:
    """高级 Ollama API 客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:11434/api", timeout: int = 60):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 的基础 URL
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        获取所有可用模型列表
        
        Returns:
            模型列表，每个模型包含名称和大小信息
        """
        try:
            response = self.session.get(
                f"{self.base_url}/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            print(f"获取模型列表失败: {e}")
            return []
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> Union[str, None]:
        """
        使用指定模型生成文本
        
        Args:
            model: 模型名称
            prompt: 提示词
            system: 系统提示词
            stream: 是否使用流式输出
            max_tokens: 最大生成的token数量
            temperature: 温度参数，控制随机性
            top_p: 控制词汇选择的多样性
            top_k: 每一步考虑的最高概率词汇数量
            repeat_penalty: 重复惩罚系数
            presence_penalty: 存在惩罚系数
            frequency_penalty: 频率惩罚系数
            stop: 停止生成的标记列表
            
        Returns:
            生成的文本或None（如果出错）
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # 添加可选参数
        if system:
            payload["system"] = system
        
        # 构建options字典
        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if presence_penalty is not None:
            options["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            options["frequency_penalty"] = frequency_penalty
        if stop:
            options["stop"] = stop
        
        # 如果有options，添加到payload
        if options:
            payload["options"] = options
        
        try:
            if stream:
                return self._handle_stream_generate(url, payload)
            else:
                return self._handle_normal_generate(url, payload)
        except Exception as e:
            print(f"生成文本时出错: {e}")
            return None
    
    def _handle_normal_generate(self, url: str, payload: Dict[str, Any]) -> Union[str, None]:
        """处理普通的生成请求"""
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            print(f"生成文本请求失败: {e}")
            return None
    
    def _handle_stream_generate(self, url: str, payload: Dict[str, Any]) -> Union[str, None]:
        """处理流式生成请求"""
        try:
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            full_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                        
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
        except requests.RequestException as e:
            print(f"流式生成请求失败: {e}")
            return None
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], str, None]:
        """
        使用聊天模式与模型交互
        
        Args:
            model: 模型名称
            messages: 消息历史列表
            stream: 是否使用流式输出
            max_tokens: 最大生成的token数量
            temperature: 温度参数，控制随机性
            top_p: 控制词汇选择的多样性
            top_k: 每一步考虑的最高概率词汇数量
            repeat_penalty: 重复惩罚系数
            presence_penalty: 存在惩罚系数
            frequency_penalty: 频率惩罚系数
            stop: 停止生成的标记列表
            
        Returns:
            聊天响应或None（如果出错）
        """
        url = f"{self.base_url}/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # 构建options字典
        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if presence_penalty is not None:
            options["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            options["frequency_penalty"] = frequency_penalty
        if stop:
            options["stop"] = stop
        
        # 如果有options，添加到payload
        if options:
            payload["options"] = options
        
        try:
            if stream:
                return self._handle_stream_chat(url, payload)
            else:
                return self._handle_normal_chat(url, payload)
        except Exception as e:
            print(f"聊天时出错: {e}")
            return None
    
    def _handle_normal_chat(self, url: str, payload: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        """处理普通的聊天请求"""
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"聊天请求失败: {e}")
            return None
    
    def _handle_stream_chat(self, url: str, payload: Dict[str, Any]) -> Union[str, None]:
        """处理流式聊天请求"""
        try:
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                        
                    line_data = json.loads(line)
                    if "message" in line_data:
                        chunk = line_data["message"].get("content", "")
                        full_content += chunk
                        print(chunk, end="", flush=True)
                except json.JSONDecodeError:
                    print(f"\n解析响应时出错: {line}")
            
            print()  # 完成后换行
            return full_content
        except requests.RequestException as e:
            print(f"流式聊天请求失败: {e}")
            return None
    
    def embeddings(self, model: str, text: str) -> Union[List[float], None]:
        """
        获取文本的嵌入向量
        
        Args:
            model: 模型名称
            text: 需要获取嵌入向量的文本
            
        Returns:
            嵌入向量列表或None（如果出错）
        """
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except requests.RequestException as e:
            print(f"获取嵌入向量失败: {e}")
            return None
    
    def interactive_chat(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        交互式聊天界面
        
        Args:
            model: 模型名称
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成的token数量
        """
        print(f"\n=== 与 {model} 进行交互式聊天 ===")
        print("输入 'exit' 或 'quit' 结束聊天")
        print("输入 'clear' 清除对话历史")
        print("输入 'save <filename>' 保存对话历史")
        print("输入 'load <filename>' 加载对话历史")
        
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
            
            if user_input.lower().startswith("save "):
                filename = user_input[5:].strip()
                self._save_chat_history(messages, filename)
                continue
            
            if user_input.lower().startswith("load "):
                filename = user_input[5:].strip()
                loaded_messages = self._load_chat_history(filename)
                if loaded_messages:
                    messages = loaded_messages
                    print(f"已加载对话历史，共 {len(messages)} 条消息")
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("\n助手: ", end="", flush=True)
            response = self.chat(
                model, 
                messages, 
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response:
                messages.append({"role": "assistant", "content": response})
            else:
                print("获取响应失败")
    
    def _save_chat_history(self, messages: List[Dict[str, str]], filename: str) -> None:
        """保存聊天历史到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到 {filename}")
        except Exception as e:
            print(f"保存对话历史失败: {e}")
    
    def _load_chat_history(self, filename: str) -> Union[List[Dict[str, str]], None]:
        """从文件加载聊天历史"""
        try:
            if not os.path.exists(filename):
                print(f"文件 {filename} 不存在")
                return None
                
            with open(filename, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            return messages
        except Exception as e:
            print(f"加载对话历史失败: {e}")
            return None
    
    def batch_generate(
        self,
        model: str,
        prompts: List[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        批量生成文本
        
        Args:
            model: 模型名称
            prompts: 提示词列表
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成的token数量
            
        Returns:
            (提示词, 生成的文本)元组列表
        """
        results = []
        total = len(prompts)
        
        print(f"开始批量处理 {total} 个提示词...")
        
        for i, prompt in enumerate(prompts):
            print(f"处理 [{i+1}/{total}]: {prompt[:50]}...")
            
            response = self.generate(
                model,
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            results.append((prompt, response or "生成失败"))
            
            # 防止API限流
            if i < total - 1:
                time.sleep(0.5)
        
        print(f"批量处理完成，成功处理 {len(results)} 个提示词")
        return results

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高级 Ollama API 客户端")
    
    # 基本参数
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="要使用的模型名称")
    parser.add_argument("--api_url", type=str, default="http://localhost:11434/api", help="Ollama API URL")
    
    # 操作模式
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chat", action="store_true", help="启动交互式聊天模式")
    group.add_argument("--prompt", type=str, help="单次生成的提示词")
    group.add_argument("--batch", action="store_true", help="批量处理模式")
    group.add_argument("--embedding", action="store_true", help="获取嵌入向量")
    
    # 生成参数
    parser.add_argument("--system", type=str, help="系统提示词")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--max_tokens", type=int, help="最大生成的token数量")
    parser.add_argument("--top_p", type=float, help="top-p 参数")
    parser.add_argument("--top_k", type=int, help="top-k 参数")
    parser.add_argument("--repeat_penalty", type=float, help="重复惩罚系数")
    
    # 流式输出
    parser.add_argument("--stream", action="store_true", help="使用流式输出")
    
    # 批处理参数
    parser.add_argument("--prompt_file", type=str, help="包含多个提示词的文件，每行一个")
    parser.add_argument("--output_file", type=str, help="批处理输出文件")
    
    # 嵌入参数
    parser.add_argument("--embedding_model", type=str, default="bge-m3", help="用于嵌入的模型")
    parser.add_argument("--text", type=str, help="要获取嵌入向量的文本")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建客户端实例
    client = OllamaClient(base_url=args.api_url)
    
    # 列出可用模型
    print("可用模型:")
    models = client.list_models()
    for model in models:
        print(f"- {model.get('name')}: {model.get('size')}")
    print()
    
    # 根据不同模式执行操作
    if args.chat:
        # 交互式聊天模式
        client.interactive_chat(
            args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    elif args.prompt:
        # 单次生成模式
        print(f"\n=== 使用 {args.model} 生成文本 ===")
        print(f"提示词: {args.prompt}")
        print("响应: ", end="" if args.stream else "\n")
        
        start_time = time.time()
        response = client.generate(
            args.model,
            args.prompt,
            system=args.system,
            stream=args.stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty
        )
        end_time = time.time()
        
        if response and not args.stream:
            print(response)
        
        print(f"\n耗时: {end_time - start_time:.2f} 秒")
    elif args.batch:
        # 批处理模式
        if not args.prompt_file:
            print("错误: 批处理模式需要指定 --prompt_file 参数")
            return
        
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取提示词文件失败: {e}")
            return
        
        print(f"\n=== 使用 {args.model} 批量生成 ===")
        start_time = time.time()
        results = client.batch_generate(
            args.model,
            prompts,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        end_time = time.time()
        
        # 输出结果
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for prompt, response in results:
                        f.write(f"提示词: {prompt}\n")
                        f.write(f"响应: {response}\n")
                        f.write("-" * 50 + "\n")
                print(f"结果已保存到 {args.output_file}")
            except Exception as e:
                print(f"保存结果失败: {e}")
        else:
            for prompt, response in results:
                print(f"提示词: {prompt}")
                print(f"响应: {response}")
                print("-" * 50)
        
        print(f"总耗时: {end_time - start_time:.2f} 秒")
    elif args.embedding:
        # 嵌入向量模式
        text = args.text or input("请输入要获取嵌入向量的文本: ")
        
        print(f"\n=== 使用 {args.embedding_model} 获取嵌入向量 ===")
        start_time = time.time()
        embedding = client.embeddings(args.embedding_model, text)
        end_time = time.time()
        
        if embedding:
            print(f"嵌入向量维度: {len(embedding)}")
            print(f"前 5 个值: {embedding[:5]}")
        
        print(f"耗时: {end_time - start_time:.2f} 秒")
    else:
        # 默认运行示例
        print("运行示例...")
        
        # 示例 1: 文本生成
        print(f"\n=== 使用 {args.model} 生成文本 ===")
        prompt = "写一首关于春天的诗"
        print(f"提示词: {prompt}")
        
        start_time = time.time()
        response = client.generate(args.model, prompt)
        end_time = time.time()
        
        if response:
            print(f"响应: {response}")
        print(f"耗时: {end_time - start_time:.2f} 秒\n")
        
        # 示例 2: 聊天完成
        print(f"\n=== 使用 {args.model} 进行聊天 ===")
        messages = [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "介绍一下你自己"}
        ]
        
        start_time = time.time()
        response = client.chat(args.model, messages)
        end_time = time.time()
        
        if response and "message" in response:
            print(f"响应: {response['message'].get('content', '')}")
        print(f"耗时: {end_time - start_time:.2f} 秒\n")
        
        # 示例 3: 流式文本生成
        print(f"\n=== 使用 {args.model} 进行流式文本生成 ===")
        prompt = "解释一下量子计算的基本原理"
        print(f"提示词: {prompt}")
        print("响应: ", end="", flush=True)
        
        start_time = time.time()
        client.generate(args.model, prompt, stream=True)
        end_time = time.time()
        
        print(f"耗时: {end_time - start_time:.2f} 秒\n")
        
        # 示例 4: 获取嵌入向量
        print(f"\n=== 使用 {args.embedding_model} 获取嵌入向量 ===")
        text = "This is a sample text for embedding"
        
        start_time = time.time()
        embeddings = client.embeddings(args.embedding_model, text)
        end_time = time.time()
        
        if embeddings:
            print(f"嵌入向量维度: {len(embeddings)}")
            print(f"前 5 个值: {embeddings[:5]}")
        else:
            print("Error getting embeddings")
        
        print(f"耗时: {end_time - start_time:.2f} 秒\n")

if __name__ == "__main__":
    main()