# Ollama API 客户端示例

这个仓库包含了三个不同复杂度的 Ollama API 客户端实现，用于与本地运行的 Ollama 模型进行交互。

## 文件说明

1. **simple_ollama_client.py**: 简单的 Ollama 客户端，使用官方推荐的方式调用 Ollama API。适合初学者了解 Ollama API 的基本用法。

2. **ollama_client.py**: 中等复杂度的 Ollama 客户端，提供了更多功能和示例，包括文本生成、聊天、流式输出和嵌入向量获取等功能。

3. **advanced_ollama_client.py**: 高级 Ollama 客户端，使用面向对象的方式实现，提供了更完善的功能，包括参数调优、交互式聊天界面、批处理和错误处理等高级特性。

## 使用要求

- Python 3.6+
- 安装了 Ollama 并运行在本地（默认地址：http://localhost:11434）
- 已安装 requests 库

## 安装依赖

```bash
pip install requests
```

## 使用方法

### 简单客户端

```bash
python simple_ollama_client.py --model qwen2.5:7b --chat
# 或者生成文本
python simple_ollama_client.py --model qwen2.5:7b --generate "写一首关于春天的诗" --stream
```

### 中等客户端

直接运行即可查看各种示例：

```bash
python ollama_client.py
```

### 高级客户端

```bash
# 交互式聊天
python advanced_ollama_client.py --model qwen2.5:7b --chat --system "你是一个有用的助手"

# 生成文本
python advanced_ollama_client.py --model qwen2.5:7b --prompt "解释量子计算的基本原理" --stream

# 批处理（需要提供包含多个提示词的文件）
python advanced_ollama_client.py --model qwen2.5:7b --batch --prompt_file prompts.txt
```

## 功能特性

- 文本生成（支持流式输出）
- 聊天对话（支持历史记录）
- 获取嵌入向量
- 参数调整（温度、top_p、top_k等）
- 批处理多个提示词
- 交互式聊天界面
- 错误处理和重试

## 注意事项

- 确保 Ollama 服务已在本地运行
- 默认使用 qwen2.5:7b 模型，可以更改为任何已安装的模型
- 如果遇到连接问题，请检查 Ollama 服务是否正常运行

## 许可证

MIT