# Running Open-Source Models from Hugging Face

## Introduction

Hugging Face has become the central hub for open-source AI models, hosting thousands of pre-trained models that you can run locally on your home AI lab. This guide covers how to discover, download, and run popular open-source models including Llama, Phi, Qwen, DeepSeek, Gemma, and many others.

Running models locally offers several advantages:
- **Privacy**: Your data never leaves your machine
- **Cost**: No API fees for inference
- **Control**: Full customization and fine-tuning capabilities
- **Speed**: No network latency for inference
- **Offline Access**: Work without internet connectivity

## Popular Open-Source Models on Hugging Face

### Large Language Models (LLMs)

#### 1. Llama (Meta)

**Models**: Llama 3.2, Llama 3.1, Llama 2
**Sizes**: 1B, 3B, 8B, 70B, 405B parameters
**License**: Llama 3 Community License

**Strengths**:
- Excellent general-purpose performance
- Strong reasoning capabilities
- Available in multiple sizes
- Extensive fine-tuned variants

**Recommended Variants**:
```
meta-llama/Llama-3.2-1B-Instruct      # Lightweight, fast
meta-llama/Llama-3.2-3B-Instruct      # Balanced performance
meta-llama/Llama-3.1-8B-Instruct      # High quality, runs on consumer GPUs
meta-llama/Llama-3.1-70B-Instruct     # Excellent performance, needs high-end GPU
```

**Hardware Requirements**:
- 1B model: 4GB VRAM (RTX 3060)
- 3B model: 8GB VRAM (RTX 3070)
- 8B model: 16GB VRAM (RTX 4080)
- 70B model: 48GB+ VRAM (A6000, A100) or CPU with 128GB+ RAM

#### 2. Phi (Microsoft)

**Models**: Phi-3, Phi-3.5
**Sizes**: Mini (3.8B), Small (7B), Medium (14B)
**License**: MIT

**Strengths**:
- Exceptional performance for size
- Optimized for efficiency
- Great for edge deployment
- Strong coding capabilities

**Recommended Variants**:
```
microsoft/Phi-3.5-mini-instruct       # 3.8B, excellent efficiency
microsoft/Phi-3-small-8k-instruct     # 7B, balanced
microsoft/Phi-3-medium-4k-instruct    # 14B, high performance
```

**Hardware Requirements**:
- Mini (3.8B): 8GB VRAM
- Small (7B): 16GB VRAM
- Medium (14B): 24GB VRAM

#### 3. Qwen (Alibaba)

**Models**: Qwen2.5, Qwen2
**Sizes**: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
**License**: Apache 2.0

**Strengths**:
- Multilingual (especially strong in Chinese)
- Excellent coding capabilities
- Strong mathematical reasoning
- Wide range of sizes

**Recommended Variants**:
```
Qwen/Qwen2.5-0.5B-Instruct           # Ultra-lightweight
Qwen/Qwen2.5-3B-Instruct             # Efficient, good quality
Qwen/Qwen2.5-7B-Instruct             # Balanced performance
Qwen/Qwen2.5-14B-Instruct            # High quality
Qwen/Qwen2.5-72B-Instruct            # Top-tier performance
```

**Hardware Requirements**:
- 0.5B: 2GB VRAM
- 3B: 8GB VRAM
- 7B: 16GB VRAM
- 14B: 24GB VRAM
- 72B: 48GB+ VRAM or CPU inference

#### 4. DeepSeek (DeepSeek AI)

**Models**: DeepSeek-V2, DeepSeek-Coder
**Sizes**: 1.3B, 6.7B, 16B, 33B, 236B
**License**: MIT

**Strengths**:
- Exceptional coding capabilities
- Strong reasoning
- Competitive with larger models
- Open weights

**Recommended Variants**:
```
deepseek-ai/deepseek-coder-1.3b-instruct    # Code-focused, lightweight
deepseek-ai/deepseek-coder-6.7b-instruct    # Excellent for coding
deepseek-ai/DeepSeek-V2-Lite                # General purpose
deepseek-ai/DeepSeek-V2                     # High performance
```

**Hardware Requirements**:
- 1.3B: 4GB VRAM
- 6.7B: 16GB VRAM
- 16B: 32GB VRAM
- 236B: Multiple GPUs or CPU

#### 5. Gemma (Google)

**Models**: Gemma 2
**Sizes**: 2B, 9B, 27B
**License**: Gemma Terms of Use

**Strengths**:
- Excellent instruction following
- Strong safety alignment
- Efficient architecture
- Good multilingual support

**Recommended Variants**:
```
google/gemma-2-2b-it                  # Lightweight, efficient
google/gemma-2-9b-it                  # Balanced performance
google/gemma-2-27b-it                 # High quality
```

**Hardware Requirements**:
- 2B: 6GB VRAM
- 9B: 18GB VRAM
- 27B: 40GB VRAM

#### 6. Mistral (Mistral AI)

**Models**: Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
**License**: Apache 2.0

**Strengths**:
- Excellent performance-to-size ratio
- Mixture of Experts (MoE) architecture
- Strong reasoning
- Fast inference

**Recommended Variants**:
```
mistralai/Mistral-7B-Instruct-v0.3    # Efficient, high quality
mistralai/Mixtral-8x7B-Instruct-v0.1  # MoE, excellent performance
mistralai/Mixtral-8x22B-Instruct-v0.1 # Top-tier MoE
```

**Hardware Requirements**:
- 7B: 16GB VRAM
- 8x7B: 32GB VRAM (only 2 experts active at once)
- 8x22B: 48GB+ VRAM

### Specialized Models

#### FoundationSec (Cisco)

**Purpose**: Cybersecurity-focused LLM
**Size**: Various
**License**: Open weights

**Use Cases**:
- Threat analysis
- Security code review
- Vulnerability detection
- Security documentation

#### Code-Specific Models

```
bigcode/starcoder2-15b                # Code generation
codellama/CodeLlama-13b-Instruct      # Meta's code model
WizardLM/WizardCoder-15B-V1.0         # Enhanced coding
```

#### Vision-Language Models

```
llava-hf/llava-1.5-7b-hf              # Image understanding
microsoft/Phi-3-vision-128k-instruct  # Multimodal Phi
```

## Installation and Setup

### Method 1: Using Transformers Library

The most flexible approach for Python developers:

```bash
# Install dependencies
pip install transformers torch accelerate bitsandbytes
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

### Method 2: Using Ollama

Ollama provides the simplest way to run models locally:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or on macOS with Homebrew
brew install ollama

# Start Ollama service
ollama serve

# Pull and run models
ollama pull llama3.2:3b
ollama run llama3.2:3b

# Pull other models
ollama pull phi3:mini
ollama pull qwen2.5:7b
ollama pull deepseek-coder:6.7b
ollama pull gemma2:9b
```

**Using Ollama with Python**:

```python
import requests
import json

def chat_with_ollama(model, messages):
    """Chat with Ollama API"""
    response = requests.post(
        'http://localhost:11434/api/chat',
        json={
            'model': model,
            'messages': messages,
            'stream': False
        }
    )
    return response.json()['message']['content']

# Example usage
messages = [
    {'role': 'user', 'content': 'Write a Python function to calculate fibonacci numbers'}
]

response = chat_with_ollama('llama3.2:3b', messages)
print(response)
```

### Method 3: Using LM Studio

LM Studio provides a GUI for running models:

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Browse and download models from the built-in browser
3. Load models with one click
4. Provides local API server compatible with OpenAI API

### Method 4: Using llama.cpp

For maximum performance and flexibility:

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Build with CUDA support (NVIDIA GPUs)
make LLAMA_CUDA=1

# Build with Metal support (Apple Silicon)
make LLAMA_METAL=1

# Download GGUF model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Run model
./main -m llama-2-7b-chat.Q4_K_M.gguf -p "Hello, how are you?" -n 128
```

### Method 5: Using Docker Model Runner

Docker Model Runner integrates with Hugging Face:

```bash
# Pull Docker Model Runner
docker pull huggingface/model-runner

# Run a model
docker run -it --gpus all \
  -p 8080:8080 \
  huggingface/model-runner \
  --model-id meta-llama/Llama-3.2-3B-Instruct

# Access via API
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is machine learning?", "parameters": {"max_new_tokens": 100}}'
```

## Quantization for Efficient Inference

Quantization reduces model size and memory requirements:

### GGUF Format (llama.cpp)

```bash
# Download quantized models from TheBloke
# Q4_K_M: 4-bit quantization, good balance
# Q5_K_M: 5-bit quantization, better quality
# Q8_0: 8-bit quantization, near-original quality

wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### GPTQ Quantization

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPTQ quantized model (4-bit)
model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)
```

### AWQ Quantization

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load AWQ quantized model
model_name = "TheBloke/Llama-2-7B-Chat-AWQ"
model = AutoAWQForCausalLM.from_quantized(
    model_name,
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### BitsAndBytes Quantization (On-the-fly)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# This 8B model now fits in ~5GB VRAM instead of 16GB!
```

## Quantization Comparison

| Method | Size Reduction | Quality Loss | Speed | Best For |
|--------|---------------|--------------|-------|----------|
| **FP16** | 50% | None | Baseline | High-end GPUs |
| **8-bit (GPTQ/AWQ)** | 75% | Minimal | Fast | Balanced |
| **4-bit (GPTQ/AWQ)** | 87.5% | Small | Very Fast | Consumer GPUs |
| **4-bit (GGUF Q4)** | 87.5% | Small | Fast | CPU/GPU |
| **3-bit (GGUF Q3)** | 90%+ | Moderate | Very Fast | Limited VRAM |

## Practical Examples

### Example 1: Running Llama 3.2 for Chat

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaChat:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.conversation_history = []
    
    def chat(self, user_message):
        """Send a message and get response"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def reset(self):
        """Clear conversation history"""
        self.conversation_history = []

# Usage
chat = LlamaChat()
print(chat.chat("What is the capital of France?"))
print(chat.chat("What's interesting about that city?"))
```

### Example 2: Code Generation with DeepSeek Coder

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_code(prompt, model_name="deepseek-ai/deepseek-coder-6.7b-instruct"):
    """Generate code using DeepSeek Coder"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.3,  # Lower temperature for code
        top_p=0.95,
        do_sample=True
    )
    
    code = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return code

# Example usage
prompt = """
Write a Python function that:
1. Takes a list of numbers
2. Removes duplicates
3. Sorts in descending order
4. Returns the top 5 numbers
Include docstring and type hints.
"""

code = generate_code(prompt)
print(code)
```

### Example 3: Multilingual with Qwen

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def multilingual_chat(text, target_language="Chinese"):
    """Chat in multiple languages using Qwen"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Respond in {target_language}."},
        {"role": "user", "content": text}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

# Example
print(multilingual_chat("Explain artificial intelligence", "Chinese"))
print(multilingual_chat("Explain artificial intelligence", "Spanish"))
```

### Example 4: Running Models with Ollama API

```python
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def list_models(self):
        """List available models"""
        response = requests.get(f"{self.base_url}/api/tags")
        return response.json()
    
    def generate(self, model, prompt, stream=False):
        """Generate text"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
        )
        return response.json()
    
    def chat(self, model, messages, stream=False):
        """Chat with model"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": stream
            }
        )
        return response.json()
    
    def pull_model(self, model):
        """Download a model"""
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                print(json.loads(line))

# Usage
client = OllamaClient()

# List available models
print(client.list_models())

# Pull a new model
client.pull_model("llama3.2:3b")

# Chat
messages = [
    {"role": "user", "content": "Write a haiku about programming"}
]
response = client.chat("llama3.2:3b", messages)
print(response['message']['content'])
```

### Example 5: Batch Processing

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def batch_inference(prompts, model_name="microsoft/Phi-3.5-mini-instruct"):
    """Process multiple prompts efficiently"""
    # Use pipeline for easy batching
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Process in batches
    results = pipe(
        prompts,
        max_new_tokens=256,
        batch_size=4,  # Adjust based on VRAM
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return [r[0]['generated_text'] for r in results]

# Example
prompts = [
    "Explain photosynthesis",
    "What is quantum entanglement?",
    "Describe the water cycle",
    "How do neural networks work?"
]

responses = batch_inference(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Performance Optimization

### 1. Flash Attention

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # 2-3x faster
)
```

### 2. Torch Compile (PyTorch 2.0+)

```python
import torch

model = AutoModelForCausalLM.from_pretrained(...)
model = torch.compile(model)  # Optimize with torch.compile
```

### 3. KV Cache Optimization

```python
# Use static KV cache for faster inference
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    use_cache=True,  # Enable KV caching
    cache_implementation="static"  # Static cache for speed
)
```

### 4. Speculative Decoding

```python
# Use a small draft model to speed up generation
from transformers import AutoModelForCausalLM

draft_model = AutoModelForCausalLM.from_pretrained("small-model")
target_model = AutoModelForCausalLM.from_pretrained("large-model")

# Speculative decoding can be 2-3x faster
outputs = target_model.generate(
    input_ids,
    assistant_model=draft_model,
    max_new_tokens=512
)
```

## Model Selection Guide

### By Hardware

| Hardware | Recommended Models |
|----------|-------------------|
| **4GB VRAM** | Qwen2.5-0.5B, Llama-3.2-1B, Phi-3.5-mini (4-bit) |
| **8GB VRAM** | Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-3B, Gemma-2-2b |
| **16GB VRAM** | Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, DeepSeek-Coder-6.7B |
| **24GB VRAM** | Qwen2.5-14B, Phi-3-medium, Gemma-2-9B, Mixtral-8x7B (4-bit) |
| **48GB+ VRAM** | Llama-3.1-70B, Qwen2.5-72B, Mixtral-8x22B |
| **CPU Only** | Any model with GGUF Q4 quantization (slower) |

### By Use Case

| Use Case | Recommended Models |
|----------|-------------------|
| **General Chat** | Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-7B |
| **Coding** | DeepSeek-Coder, CodeLlama, Qwen2.5-Coder |
| **Multilingual** | Qwen2.5, Llama-3.1, Gemma-2 |
| **Math/Reasoning** | Qwen2.5, DeepSeek-V2, Llama-3.1 |
| **Edge Deployment** | Phi-3.5-mini, Qwen2.5-0.5B, Gemma-2-2b |
| **Research** | Llama-3.1-70B, Qwen2.5-72B, Mixtral-8x22B |

## Troubleshooting

### Out of Memory Errors

```python
# Solution 1: Use quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

# Solution 2: Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)

# Solution 3: Reduce max_length
outputs = model.generate(input_ids, max_new_tokens=128)  # Instead of 512
```

### Slow Generation

```python
# Enable optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16
    device_map="auto",
    attn_implementation="flash_attention_2"  # Flash attention
)

# Use torch.compile
model = torch.compile(model, mode="reduce-overhead")

# Adjust generation parameters
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    num_beams=1,  # Use greedy decoding instead of beam search
    do_sample=True
)
```

## Conclusion

Running open-source models locally has never been easier. With tools like Hugging Face Transformers, Ollama, and llama.cpp, you can run state-of-the-art models on consumer hardware. The key is choosing the right model for your hardware and use case, and applying appropriate optimizations like quantization.

## Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Ollama Library](https://ollama.com/library)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [TheBloke's Quantized Models](https://huggingface.co/TheBloke)

