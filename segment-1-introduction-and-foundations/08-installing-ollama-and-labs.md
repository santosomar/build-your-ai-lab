# Installing Ollama

## Introduction

Ollama is a powerful tool that makes it easy to run large language models (LLMs) locally on your machine. It simplifies the process of downloading, managing, and running models like Llama, Mistral, Phi, and many others.

## What is Ollama?

**Ollama** is an open-source application that:
- Runs LLMs locally on your hardware
- Manages model downloads and storage
- Provides a simple API for interaction
- Supports multiple models
- Optimizes performance automatically
- Works on CPU and GPU

**Key Features:**
- Easy installation and setup
- Simple command-line interface
- REST API for integration
- Automatic model quantization
- GPU acceleration (NVIDIA, AMD, Apple Metal)
- Model library with popular LLMs

## System Requirements

### Minimum Requirements

**For Small Models (< 7B parameters):**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 10GB free
- GPU: Optional (CPU works)

**For Medium Models (7B-13B parameters):**
- CPU: 6+ cores
- RAM: 16GB+
- Storage: 20GB free
- GPU: 8GB+ VRAM (recommended)

**For Large Models (30B-70B parameters):**
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 50GB+ free
- GPU: 24GB+ VRAM (required for good performance)

### Supported Platforms

- Linux (x86_64, ARM64)
  - x86_64: Traditional Intel/AMD systems
  - ARM64: Raspberry Pi, NVIDIA DGX Spark, cloud ARM instances
- macOS (Intel and Apple Silicon)
- Windows (with WSL2 or native)

### GPU Support

**NVIDIA:**
- CUDA 11.2 or higher
- Driver version 470.57.02+
- Automatic detection and use

**AMD:**
- ROCm support (Linux)
- Limited compared to NVIDIA

**Apple Silicon:**
- Metal acceleration
- Unified memory advantage
- Good performance on M1/M2/M3

**NVIDIA DGX Spark (Arm-based):**
- CUDA acceleration via Blackwell GPU
- 128GB unified memory
- Excellent performance for AI workloads
- Native ARM64 support

## Installation
Go to [Ollama Official Website](https://ollama.com) for the latest installation instructions.

## Labs
- [Omar's Ollama Labs - Hands-on Tutorials](https://github.com/The-Art-of-Hacking/h4cker/tree/master/ai_research/ollama-labs) - Comprehensive collection of practical labs and exercises for learning Ollama

## Basic Usage

### Running Your First Model

```bash
# Run a model (downloads if not present)
ollama run llama3.2

# This will:
# 1. Download the model (first time only)
# 2. Load it into memory
# 3. Start an interactive chat session

# Example interaction:
>>> Hello! How are you?
I'm doing well, thank you for asking! How can I help you today?

>>> /bye
# Exit the chat
```

### Popular Models to Try

```bash
# Llama 3.2 (Meta) - 3B parameters
ollama run gpt-oss

# Phi 3 (Microsoft) - 3.8B parameters
ollama run phi4

# Gemma 2 (Google) - 9B parameters
ollama run gemma3

# Qwen 3 (Alibaba) - 7B parameters
ollama run qwen3

# Qwen 3 Coder (Alibaba) - 7B parameters
ollama run qwen3-coder
```

### Model Management

```bash
# List downloaded models
ollama list

# Pull a model without running
ollama pull llama3.2

# Remove a model
ollama rm llama3.2

# Show model information
ollama show llama3.2

# Show model details (layers, parameters, etc.)
ollama show llama3.2 --modelfile
```

### Model Sizes and Variants

Models come in different sizes and quantizations:

```bash
# Different sizes of Llama 3.1
ollama pull llama3.1:8b      # 8 billion parameters (default)
ollama pull llama3.1:70b     # 70 billion parameters
ollama pull llama3.1:405b    # 405 billion parameters (requires significant resources)

# Different quantizations (smaller = less VRAM, faster, less accurate)
ollama pull llama3.1:8b-q4_0    # 4-bit quantization
ollama pull llama3.1:8b-q8_0    # 8-bit quantization
```

**Quantization Guide:**
- `q4_0` - 4-bit, smallest, fastest, lowest quality
- `q5_0` - 5-bit, good balance
- `q8_0` - 8-bit, larger, slower, better quality
- No suffix - Default quantization (usually q4_0)

## Advanced Usage

### Command-Line Options

```bash
# Run with specific prompt
ollama run llama3.2 "Explain quantum computing in simple terms"

# Run with system prompt
ollama run llama3.2 --system "You are a helpful coding assistant"

# Set temperature (0.0 = deterministic, 1.0 = creative)
ollama run llama3.2 --temperature 0.7

# Set context window size
ollama run llama3.2 --context-length 4096
```

### Interactive Commands

While in a chat session:

```bash
# Show help
>>> /?

# Load image (for multimodal models)
>>> /image path/to/image.jpg

# Set system prompt
>>> /system You are a Python expert

# Show current settings
>>> /show

# Exit
>>> /bye
```

### Using the API

Ollama provides a REST API on `http://localhost:11434`

**Generate Completion:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

**Chat Completion:**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {
      "role": "user",
      "content": "Hello! How are you?"
    }
  ],
  "stream": false
}'
```

**Python Example:**
```python
import requests
import json

def chat_with_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()['response']

# Usage
response = chat_with_ollama("Explain machine learning in one sentence")
print(response)
```

**Streaming Response:**
```python
import requests
import json

def stream_chat(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get('done'):
                    print(chunk['response'], end='', flush=True)
                else:
                    print()  # New line at end

# Usage
stream_chat("Write a haiku about AI")
```

### Python Library

```bash
# Install official Python library
pip install ollama
```

```python
import ollama

# Simple chat
response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'Why is the ocean salty?',
    },
])
print(response['message']['content'])

# Streaming chat
stream = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Tell me a joke'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Generate embeddings
embeddings = ollama.embeddings(
    model='llama3.2',
    prompt='The quick brown fox jumps over the lazy dog',
)
print(embeddings['embedding'])

# List models
models = ollama.list()
for model in models['models']:
    print(model['name'])
```

## Creating Custom Models

### Modelfile

Create custom models with specific behaviors:

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2

# Set temperature
PARAMETER temperature 0.8

# Set system message
SYSTEM You are a helpful AI assistant specialized in Python programming. You provide clear, concise code examples and explanations.

# Set context window
PARAMETER num_ctx 4096
EOF

# Create custom model
ollama create python-assistant -f Modelfile

# Run custom model
ollama run python-assistant
```

### Fine-tuning Adapters

```bash
# Create model with LoRA adapter
cat > Modelfile << 'EOF'
FROM llama3.2
ADAPTER ./path/to/lora-adapter.bin
EOF

ollama create custom-model -f Modelfile
```

## Performance Optimization

### GPU Configuration

```bash
# Check GPU usage
nvidia-smi

# Set specific GPU (if multiple)
CUDA_VISIBLE_DEVICES=0 ollama serve

# Limit GPU memory
OLLAMA_GPU_MEMORY_FRACTION=0.8 ollama serve
```

### Memory Management

```bash
# Set number of models to keep in memory
OLLAMA_NUM_PARALLEL=2 ollama serve

# Set keep-alive time (how long to keep model loaded)
OLLAMA_KEEP_ALIVE=5m ollama serve

# Unload model immediately after use
OLLAMA_KEEP_ALIVE=0 ollama serve
```

### CPU Optimization

```bash
# Set number of CPU threads
OLLAMA_NUM_THREADS=8 ollama serve

# For CPU-only systems
OLLAMA_COMPUTE_UNIT=cpu ollama serve
```

## Troubleshooting

### Model Download Issues

```bash
# Check connection
curl -I https://ollama.com

# Manual download location
# Linux: ~/.ollama/models
# macOS: ~/.ollama/models
# Windows: C:\Users\<username>\.ollama\models

# Clear cache and re-download
rm -rf ~/.ollama/models/manifests
ollama pull llama3.2
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Check Ollama logs
sudo journalctl -u ollama -f
```

### Out of Memory

```bash
# Try smaller model
ollama run llama3.2:3b

# Try quantized version
ollama run llama3.2:8b-q4_0

# Reduce context length
ollama run llama3.2 --context-length 2048

# Close other applications
# Monitor memory usage
htop  # or top
```

### Slow Performance

```bash
# Check if using GPU
ollama ps  # Shows running models and device

# Reduce batch size
OLLAMA_BATCH_SIZE=256 ollama serve

# Use smaller model
ollama run phi3  # Smaller, faster

# Check system resources
nvidia-smi  # GPU
htop        # CPU/RAM
```

## Integration Examples

### LangChain Integration

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")

# Simple query
response = llm.invoke("What is the capital of France?")
print(response)

# With prompt template
from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = prompt | llm

response = llm_chain.invoke({"question": "What is 25 * 4?"})
print(response)
```

### Jupyter Notebook Integration

```python
# In Jupyter notebook
import ollama

def ask_ollama(question, model="llama3.2"):
    """Helper function for Jupyter"""
    response = ollama.chat(model=model, messages=[
        {'role': 'user', 'content': question}
    ])
    return response['message']['content']

# Usage
answer = ask_ollama("Explain gradient descent")
print(answer)
```

## Best Practices

### Model Selection

1. **Start small** - Begin with 3B-7B parameter models
2. **Test performance** - Benchmark on your hardware
3. **Consider use case** - Code, chat, analysis, etc.
4. **Balance quality/speed** - Larger isn't always better

### Resource Management

1. **Monitor usage** - Watch GPU/CPU/RAM
2. **Unload when done** - Set OLLAMA_KEEP_ALIVE appropriately
3. **Use quantization** - q4_0 for most tasks
4. **Batch requests** - More efficient than one-by-one

### Security

1. **Local network only** - Don't expose to internet without security
2. **Use firewall** - Restrict port 11434
3. **Validate inputs** - Sanitize user prompts
4. **Monitor logs** - Check for unusual activity

## Resources

- [Ollama Official Website](https://ollama.com)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Model Library](https://ollama.com/library)
- [Ollama Discord](https://discord.gg/ollama)
- [Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [Omar's Ollama Labs - Hands-on Tutorials](https://github.com/The-Art-of-Hacking/h4cker/tree/master/ai_research/ollama-labs) - Comprehensive collection of practical labs and exercises for learning Ollama

## Next Steps

- **[AI Frameworks](./09-ai-frameworks.md)** - TensorFlow, PyTorch, Hugging Face
- **[Security and Network Setup](./10-security-network.md)** - Secure your lab
- **[Segment 3](../segment-3-integrating-and-leveraging-ai-environments/)** - Running open-source models



