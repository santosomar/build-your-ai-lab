# Do You Need to Install AI Frameworks? (TensorFlow, PyTorch, Hugging Face)

## Introduction

AI frameworks are powerful libraries that simplify building, training, and deploying machine learning models. But do you actually need them? This guide helps you understand when to install these frameworks and which ones to choose.

## Quick Answer

**Do you need AI frameworks?**

- **Using Ollama only?** → No, Ollama handles everything
- **Fine-tuning models?** → Yes, you'll need PyTorch or TensorFlow
- **Building custom models?** → Yes, definitely
- **Using pre-trained models?** → Maybe, Hugging Face makes it easy
- **Traditional ML only?** → No, scikit-learn is sufficient
- **Learning AI/ML?** → Yes, essential for understanding

## Overview of Major Frameworks

### PyTorch

**What is it?**
- Deep learning framework by Meta (Facebook)
- Dynamic computational graphs
- Pythonic and intuitive
- Research-focused but production-ready

**Pros:**
- Easy to learn and debug
- Flexible and dynamic
- Great for research
- Excellent documentation
- Large community
- Native Python feel

**Cons:**
- Historically weaker deployment tools (improving)
- Can be slower than TensorFlow for production
- More verbose for simple models

**Best For:**
- Research and experimentation
- Custom architectures
- Learning deep learning
- Computer vision
- NLP with Transformers

### TensorFlow

**What is it?**
- Deep learning framework by Google
- Static computational graphs (eager execution now available)
- Production-focused
- Comprehensive ecosystem

**Pros:**
- Excellent deployment tools (TF Lite, TF Serving)
- Better for production at scale
- TensorBoard visualization
- Mobile and edge deployment
- TPU support
- Keras integration (high-level API)

**Cons:**
- Steeper learning curve
- Less intuitive than PyTorch
- Debugging can be harder
- More boilerplate code

**Best For:**
- Production deployment
- Mobile/edge AI
- Large-scale training
- TPU usage
- Keras users

### Hugging Face Transformers

**What is it?**
- Library for pre-trained transformer models
- Built on PyTorch and TensorFlow
- Focuses on NLP and multimodal models
- Largest model hub

**Pros:**
- Access to thousands of pre-trained models
- Extremely easy to use
- Consistent API
- Active community
- Regular updates
- Great documentation

**Cons:**
- Focused on transformers (not general-purpose)
- Can be heavy (large dependencies)
- Abstracts away details (good and bad)

**Best For:**
- Using pre-trained LLMs
- NLP tasks
- Quick prototyping
- Transfer learning
- Fine-tuning models

### JAX

**What is it?**
- NumPy on steroids by Google
- Automatic differentiation
- JIT compilation
- Designed for high performance

**Pros:**
- Very fast
- Functional programming style
- Great for research
- Excellent for numerical computing
- TPU support

**Cons:**
- Steeper learning curve
- Smaller community
- Fewer pre-built models
- Less beginner-friendly

**Best For:**
- High-performance computing
- Research (especially RL)
- Custom algorithms
- Numerical optimization

## Decision Tree

```
What do you want to do?

├─ Use existing LLMs locally
│  └─ Use Ollama (no framework needed)
│
├─ Fine-tune existing models
│  ├─ NLP/LLMs → PyTorch + Hugging Face
│  ├─ Computer Vision → PyTorch or TensorFlow
│  └─ General → PyTorch (easier) or TensorFlow (production)
│
├─ Build custom models from scratch
│  ├─ Research/Learning → PyTorch
│  ├─ Production → TensorFlow
│  └─ High Performance → JAX
│
├─ Deploy to mobile/edge
│  └─ TensorFlow Lite or PyTorch Mobile
│
├─ Traditional ML (not deep learning)
│  └─ scikit-learn (no deep learning framework needed)
│
└─ Just learning/exploring
   └─ PyTorch (most intuitive)
```

## Installation Guide

### PyTorch

**Check Requirements:**
- Python 3.8+
- NVIDIA GPU (optional but recommended)
- CUDA 11.8 or 12.1 (for GPU)
- **Note:** For Arm-based systems (NVIDIA DGX Spark, Apple Silicon), ensure you use ARM64-compatible builds

**Installation:**

Visit [pytorch.org](https://pytorch.org) for the latest installation command.

```bash
# Activate your environment
conda activate ai-lab

# NVIDIA GPU (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# NVIDIA GPU (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Apple Silicon (Mac M1/M2/M3) and Arm systems
conda install pytorch torchvision torchaudio -c pytorch

# NVIDIA DGX Spark (Arm + CUDA)
# Use CUDA-enabled ARM64 builds
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Using pip (GPU)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Using pip (CPU)
pip3 install torch torchvision torchaudio
```

**Verify Installation:**

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Test tensor operation
x = torch.rand(5, 3)
print(f"\nTest tensor:\n{x}")

# Test GPU operation
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"Tensor on GPU: {x_gpu.device}")
```

### TensorFlow

**Check Requirements:**
- Python 3.9-3.11 (check compatibility)
- NVIDIA GPU (optional)
- CUDA 11.8 and cuDNN 8.6 (for GPU)

**Installation:**

```bash
# Activate your environment
conda activate ai-lab

# GPU version (includes CPU)
pip install tensorflow[and-cuda]

# CPU only
pip install tensorflow

# macOS (Apple Silicon)
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration

# Specific version
pip install tensorflow==2.15.0
```

**Verify Installation:**

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List devices
print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Test operation
x = tf.random.normal([5, 3])
print(f"\nTest tensor:\n{x}")

# Test GPU
if len(tf.config.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        y = tf.random.normal([5, 3])
        print(f"Tensor on GPU: {y.device}")
```

### Hugging Face Transformers

**Installation:**

```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers[torch]

# With TensorFlow
pip install transformers[tf]

# Full installation (all features)
pip install transformers[all]

# Additional useful libraries
pip install datasets  # For datasets
pip install accelerate  # For training optimization
pip install sentencepiece  # For some tokenizers
pip install protobuf  # For some models
```

**Verify Installation:**

```python
import transformers
from transformers import pipeline

print(f"Transformers version: {transformers.__version__}")

# Test with a simple pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
print(f"\nSentiment analysis test: {result}")

# List available pipelines
print("\nAvailable pipeline tasks:")
print("- text-classification")
print("- token-classification")
print("- question-answering")
print("- text-generation")
print("- summarization")
print("- translation")
print("- and many more...")
```

## Minimal Installation for Different Use Cases

### Use Case 1: Just Using Ollama

```bash
# No frameworks needed!
# Just install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Optional: Python client
pip install ollama
```

### Use Case 2: Fine-tuning LLMs

```bash
# Create environment
conda create -n finetune python=3.11 -y
conda activate finetune

# Install PyTorch (GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Hugging Face ecosystem
pip install transformers datasets accelerate peft bitsandbytes

# Optional: Training tools
pip install wandb tensorboard
```

### Use Case 3: Computer Vision

```bash
# Create environment
conda create -n vision python=3.11 -y
conda activate vision

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install vision libraries
pip install opencv-python pillow albumentations

# Optional: Pre-trained models
pip install timm  # PyTorch Image Models
```

### Use Case 4: Traditional ML

```bash
# Create environment
conda create -n ml python=3.11 -y
conda activate ml

# No deep learning frameworks needed!
conda install scikit-learn pandas numpy matplotlib seaborn

# Optional: Additional ML tools
pip install xgboost lightgbm catboost
```

### Use Case 5: Learning/Experimentation

```bash
# Create environment
conda create -n learning python=3.11 -y
conda activate learning

# Install PyTorch (easier to learn)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Jupyter and visualization
conda install jupyterlab matplotlib seaborn

# Install Hugging Face for easy access to models
pip install transformers datasets

# Add kernel to Jupyter
python -m ipykernel install --user --name=learning
```

## Framework Comparison

### Training a Simple Neural Network

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model, loss, optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

**TensorFlow/Keras:**
```python
import tensorflow as tf
from tensorflow import keras

# Define model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(inputs, labels, epochs=10)
```

**Verdict:** TensorFlow/Keras is more concise for simple models, PyTorch is more explicit and flexible.

## Common Issues and Solutions

### CUDA Out of Memory

```python
# Reduce batch size
batch_size = 16  # Try 8, 4, or 2

# Clear cache (PyTorch)
torch.cuda.empty_cache()

# Use gradient accumulation
# Instead of batch_size=64, use batch_size=16 with 4 accumulation steps

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Import Errors

```bash
# Verify environment
conda env list
conda activate your-env

# Reinstall package
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check Python path
python -c "import sys; print(sys.executable)"
```

### Version Conflicts

```bash
# Create fresh environment
conda create -n fresh python=3.11 -y
conda activate fresh

# Install in correct order
# 1. PyTorch/TensorFlow first
# 2. Then other libraries

# Check versions
pip list | grep torch
pip list | grep tensorflow
```

## Do You Really Need Them?

### You DON'T Need Frameworks If:

✅ Only using Ollama for LLM inference
✅ Only doing traditional ML (use scikit-learn)
✅ Only using pre-built APIs (OpenAI, Anthropic, etc.)
✅ Only deploying pre-trained models via API
✅ Using no-code/low-code AI tools

### You DO Need Frameworks If:

✅ Fine-tuning models
✅ Training custom models
✅ Learning deep learning
✅ Research and experimentation
✅ Custom architectures
✅ Production ML pipelines
✅ Computer vision or NLP projects

## Recommendations

### For Beginners

**Start with:**
1. Ollama (for LLM experimentation)
2. scikit-learn (for traditional ML)
3. PyTorch (when ready for deep learning)

**Why:**
- Ollama is easiest to get started
- scikit-learn teaches ML fundamentals
- PyTorch is most intuitive for learning

### For Researchers

**Use:**
- PyTorch (primary)
- Hugging Face (for NLP)
- JAX (for cutting-edge research)

**Why:**
- Flexibility and control
- Latest research uses PyTorch
- Easy to implement papers

### For Production

**Use:**
- TensorFlow (deployment)
- PyTorch (development)
- Hugging Face (pre-trained models)

**Why:**
- TensorFlow has better deployment tools
- PyTorch for development flexibility
- Hugging Face for quick integration

### For Hobbyists

**Start with:**
- Ollama (immediate results)
- Hugging Face (easy pre-trained models)
- PyTorch (if building custom models)

**Why:**
- Quick wins and motivation
- Less setup frustration
- Learn progressively

## Next Steps

- **[Security and Network Setup](./10-security-network.md)** - Secure your AI lab
- **[Segment 2: Cloud-Based AI Labs](../segment-2-cloud-based-ai-labs/)** - Explore cloud options
- **[Segment 3: Integrating AI Environments](../segment-3-integrating-and-leveraging-ai-environments/)** - Hybrid approaches

## Resources

- PyTorch Tutorials - https://pytorch.org/tutorials/
- TensorFlow Tutorials - https://www.tensorflow.org/tutorials
- Hugging Face Course - https://huggingface.co/learn
- Fast.ai - https://www.fast.ai/ (PyTorch-based course)
- Deep Learning Book - https://www.deeplearningbook.org/

