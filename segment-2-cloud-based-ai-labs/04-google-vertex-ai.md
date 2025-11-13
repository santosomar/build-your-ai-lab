# Google Vertex AI

## Introduction

Google Vertex AI is Google Cloud's unified machine learning platform that brings together all Google Cloud services for building, deploying, and scaling ML models. It provides access to Google's powerful Gemini models and comprehensive MLOps tools.

## What is Vertex AI?

**Vertex AI** is Google's comprehensive AI platform offering:
- Gemini foundation models (1.5 Pro, Flash, Ultra)
- Custom model training and AutoML
- Model Garden (pre-trained models)
- Vertex AI Workbench (notebooks)
- MLOps and deployment tools
- Generative AI Studio

**Key Strengths:**
- Gemini models with massive context (up to 2M tokens)
- TPU access for TensorFlow workloads
- Tight integration with Google Cloud services
- Competitive pricing
- Strong data analytics integration (BigQuery)

## Core Components

### 1. Gemini Models

**Gemini 1.5 Pro:**
- **Context:** 2 million tokens (largest available)
- **Capabilities:** Multimodal (text, images, video, audio, code)
- **Strengths:** Long-context understanding, complex reasoning
- **Pricing:** 
  - ≤128K tokens: $0.00125/1K input, $0.005/1K output
  - >128K tokens: $0.01/1K input, $0.03/1K output

**Gemini 1.5 Flash:**
- **Context:** 1 million tokens
- **Capabilities:** Multimodal, optimized for speed
- **Strengths:** Fast, cost-effective, good for high-volume
- **Pricing:** $0.000075/1K input, $0.0003/1K output

**Gemini 1.0 Ultra:**
- **Context:** 128K tokens
- **Capabilities:** Most capable for complex tasks
- **Strengths:** Advanced reasoning, coding
- **Pricing:** $0.0125/1K input, $0.0375/1K output

**Use Cases:**
- **Gemini 1.5 Pro:** Long document analysis, video understanding, complex reasoning
- **Gemini 1.5 Flash:** High-volume applications, real-time chat, quick tasks
- **Gemini 1.0 Ultra:** Most demanding tasks requiring highest capability

### 2. Vertex AI Studio

**What:** Web-based interface for generative AI

**Features:**
- **Prompt Design:** Test and iterate on prompts
- **Multimodal Prompts:** Text, images, video
- **Model Tuning:** Fine-tune models on your data
- **Model Garden:** Browse and deploy pre-trained models
- **Safety Settings:** Content filtering

**Getting Started:**
```python
from vertexai.preview.generative_models import GenerativeModel

model = GenerativeModel('gemini-1.5-pro')

response = model.generate_content(
    "Explain quantum computing in simple terms"
)

print(response.text)
```

### 3. Vertex AI Workbench

**What:** Managed Jupyter notebook environment

**Features:**
- Pre-configured for ML (TensorFlow, PyTorch, JAX)
- GPU/TPU support
- Git integration
- BigQuery integration
- Scheduled execution

**Instance Types:**
- n1-standard-4 (4 vCPU, 15GB RAM): $0.19/hour
- n1-highmem-8 (8 vCPU, 52GB RAM): $0.47/hour
- a2-highgpu-1g (1 A100 GPU): $3.67/hour
- v3-8 (TPU v3): $8.00/hour

### 4. AutoML

**What:** Automated machine learning

**Supported Tasks:**
- **AutoML Tables:** Structured data (classification, regression)
- **AutoML Vision:** Image classification, object detection
- **AutoML Natural Language:** Text classification, entity extraction
- **AutoML Video:** Video classification, action recognition

**Process:**
1. Upload training data
2. Select objective (e.g., maximize accuracy)
3. AutoML trains multiple models
4. Evaluate and deploy best model

**Pricing:** Based on training hours and node hours

### 5. Custom Training

**Training Options:**
- **Pre-built Containers:** TensorFlow, PyTorch, Scikit-learn, XGBoost
- **Custom Containers:** Bring your own
- **Distributed Training:** Multi-GPU, multi-node

**Example:**

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name='my-training-job',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['pandas', 'scikit-learn'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest'
)

model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

### 6. Model Garden

**What:** Repository of pre-trained models

**Available Models:**
- Google models (Gemini, PaLM, Imagen)
- Open-source models (Llama, Mistral, Falcon)
- Partner models (Anthropic Claude)
- Task-specific models (vision, NLP, speech)

**Deployment:**
- One-click deployment
- Customizable endpoints
- Auto-scaling

### 7. Vertex AI Pipelines

**What:** Orchestrate ML workflows

**Features:**
- Kubeflow Pipelines integration
- TFX (TensorFlow Extended) support
- Automated workflows
- Version control
- Scheduling

**Example:**

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component

@component
def preprocess_data(input_path: str, output_path: str):
    # Preprocessing logic
    pass

@component
def train_model(data_path: str, model_path: str):
    # Training logic
    pass

@dsl.pipeline(name='my-pipeline')
def pipeline(input_path: str):
    preprocess_task = preprocess_data(input_path=input_path)
    train_task = train_model(data_path=preprocess_task.outputs['output_path'])

from kfp.v2 import compiler
compiler.Compiler().compile(pipeline_func=pipeline, package_path='pipeline.json')

# Submit pipeline
from google.cloud import aiplatform
job = aiplatform.PipelineJob(
    display_name='my-pipeline-run',
    template_path='pipeline.json',
    parameter_values={'input_path': 'gs://my-bucket/data'}
)
job.submit()
```

### 8. Vertex AI Endpoints

**Deployment Options:**

**Online Prediction:**
- Real-time inference
- Auto-scaling
- A/B testing
- Traffic splitting

**Batch Prediction:**
- Process large datasets
- Cost-effective
- Asynchronous

**Example:**

```python
# Deploy model
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)

# Make prediction
prediction = endpoint.predict(instances=[[1.0, 2.0, 3.0, 4.0]])
print(prediction)
```

### 9. Vertex AI Feature Store

**What:** Centralized feature repository

**Benefits:**
- Feature sharing across projects
- Online and offline serving
- Point-in-time correctness
- Feature monitoring

**Pricing:**
- Online serving: $0.35 per million reads
- Offline serving: $0.05 per GB scanned
- Storage: $0.025 per GB-month

### 10. Vertex AI Model Monitoring

**What:** Monitor deployed models

**Monitors:**
- Prediction drift
- Training-serving skew
- Feature attribution
- Custom metrics

**Alerts:**
- Email notifications
- Cloud Monitoring integration
- Automated actions

## Getting Started

### Prerequisites

1. Google Cloud account
2. Enable Vertex AI API
3. Set up billing
4. Create service account

### Step 1: Setup

```bash
# Install SDK
pip install google-cloud-aiplatform

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project my-project
```

### Step 2: Use Gemini API

```python
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

# Initialize
vertexai.init(project='my-project', location='us-central1')

# Create model
model = GenerativeModel('gemini-1.5-pro')

# Simple text generation
response = model.generate_content("Write a poem about AI")
print(response.text)

# Multimodal (text + image)
image_part = Part.from_uri(
    uri='gs://my-bucket/image.jpg',
    mime_type='image/jpeg'
)

response = model.generate_content([
    "What's in this image?",
    image_part
])
print(response.text)

# Streaming
responses = model.generate_content(
    "Tell me a long story",
    stream=True
)

for response in responses:
    print(response.text, end='')
```

### Step 3: Use with LangChain

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model_name='gemini-1.5-pro',
    temperature=0.7,
    max_output_tokens=2048
)

response = llm.invoke("Explain machine learning")
print(response.content)
```

### Step 4: RAG with Vertex AI

```python
from vertexai.preview.generative_models import GenerativeModel, grounding

model = GenerativeModel('gemini-1.5-pro')

# Use Google Search grounding
response = model.generate_content(
    "What are the latest developments in quantum computing?",
    generation_config={'temperature': 0.2},
    grounding_source=grounding.GoogleSearchRetrieval()
)

print(response.text)
print(response.grounding_metadata)  # Sources
```

## Best Practices

### 1. Cost Optimization

**Strategies:**
- Use Gemini 1.5 Flash for high-volume tasks
- Leverage context caching (coming soon)
- Use batch prediction for bulk processing
- Preemptible VMs for training (60-90% discount)
- Committed use discounts (30-70% savings)

### 2. Long Context Usage

**Gemini 1.5 Pro Tips:**
- Entire codebases (up to 2M tokens)
- Long documents (books, research papers)
- Video analysis (up to 1 hour)
- Conversation history

**Example:**

```python
# Analyze entire codebase
with open('large_file.txt', 'r') as f:
    content = f.read()  # Can be up to 2M tokens

response = model.generate_content([
    "Analyze this codebase and suggest improvements:",
    content
])
```

### 3. Multimodal Applications

```python
# Video understanding
video_part = Part.from_uri(
    uri='gs://my-bucket/video.mp4',
    mime_type='video/mp4'
)

response = model.generate_content([
    "Summarize this video and identify key moments",
    video_part
])

# Audio transcription and analysis
audio_part = Part.from_uri(
    uri='gs://my-bucket/audio.mp3',
    mime_type='audio/mpeg'
)

response = model.generate_content([
    "Transcribe this audio and identify the speakers",
    audio_part
])
```

## Use Cases

### 1. Long Document Analysis
- Legal document review
- Research paper analysis
- Technical documentation
- Book summarization

### 2. Video Understanding
- Content moderation
- Video summarization
- Scene detection
- Action recognition

### 3. Code Analysis
- Codebase understanding
- Bug detection
- Code review
- Documentation generation

### 4. Multimodal Applications
- Image + text understanding
- Video + audio analysis
- Document + image extraction

## Comparison: Vertex AI vs. Competitors

| Feature | Vertex AI | AWS (Bedrock/SageMaker) | Azure (AI Foundry) |
|---------|-----------|------------------------|-------------------|
| **Foundation Models** | Gemini, PaLM | Claude, Llama, Titan | GPT-4, GPT-4o |
| **Max Context** | 2M tokens | 200K tokens | 128K tokens |
| **Multimodal** | Text, image, video, audio | Text, image | Text, image |
| **TPU Access** | Yes | No | No |
| **AutoML** | Strong | Autopilot | AutoML |
| **Pricing** | Competitive | Varies | Similar to OpenAI |
| **Best For** | Google ecosystem, long context | AWS ecosystem | Microsoft ecosystem |

## Limitations

**Current Limitations:**
- Gemini availability varies by region
- Some features in preview
- Rate limits (vary by model and region)
- TPUs primarily for TensorFlow

**Considerations:**
- Learning curve for Google Cloud
- Pricing complexity
- Regional availability

## Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Reference](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/)
- [Vertex AI Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples)
- [Generative AI Learning Path](https://cloud.google.com/learn/training/machinelearning-ai)

## Next Steps

- **[Azure AI Foundry](./05-azure-ai-foundry.md)** - Microsoft's AI platform
- **[OpenAI Agent Builder](./06-openai-agent-builder.md)** - Building AI agents
- **[Cost Management](./07-cost-management.md)** - Optimize cloud spending



