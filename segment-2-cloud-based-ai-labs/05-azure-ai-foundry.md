# Microsoft Azure AI Foundry and AI Foundry Agent Service

## Introduction

Azure AI Foundry (formerly Azure AI Studio) is Microsoft's unified platform for building, evaluating, and deploying AI applications. It provides access to OpenAI models, open-source models, and Microsoft's own AI services, with deep integration into the Microsoft ecosystem.

## What is Azure AI Foundry?

**Azure AI Foundry** is Microsoft's comprehensive AI development platform offering:
- Azure OpenAI Service (GPT-4, GPT-4o, GPT-3.5, DALL-E 3)
- Model catalog (Meta Llama, Mistral, Cohere, etc.)
- Prompt flow for orchestration
- Content safety and responsible AI tools
- AI Foundry Agent Service for building agents
- Integration with Microsoft 365, Dynamics, Power Platform

**Key Strengths:**
- Exclusive access to OpenAI models in enterprise environment
- Strong Microsoft ecosystem integration
- Enterprise-grade security and compliance
- Hybrid cloud capabilities
- Responsible AI tools built-in

## Core Components

### 1. Azure OpenAI Service

**Available Models:**

**GPT-4o (Omni):**
- **Context:** 128K tokens
- **Capabilities:** Multimodal (text, images), fast
- **Pricing:** $0.005/1K input, $0.015/1K output
- **Best For:** General-purpose, multimodal tasks

**GPT-4 Turbo:**
- **Context:** 128K tokens
- **Capabilities:** Advanced reasoning, JSON mode
- **Pricing:** $0.01/1K input, $0.03/1K output
- **Best For:** Complex tasks, structured output

**GPT-3.5 Turbo:**
- **Context:** 16K tokens
- **Capabilities:** Fast, cost-effective
- **Pricing:** $0.0005/1K input, $0.0015/1K output
- **Best For:** Simple tasks, high-volume

**DALL-E 3:**
- **Capabilities:** High-quality image generation
- **Pricing:** $0.04 per image (1024x1024)
- **Best For:** Creative image generation

**Whisper:**
- **Capabilities:** Speech-to-text
- **Pricing:** $0.006 per minute
- **Best For:** Transcription, voice interfaces

**Text Embedding Models:**
- text-embedding-3-large: $0.00013/1K tokens
- text-embedding-3-small: $0.00002/1K tokens

### 2. Azure AI Foundry Hub and Projects

**Hub:**
- Centralized resource management
- Shared connections and compute
- Team collaboration
- Governance and compliance

**Projects:**
- Individual workspaces
- Model deployments
- Prompt flows
- Evaluations

**Setup:**

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Create client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id='your-subscription-id',
    resource_group_name='your-resource-group',
    workspace_name='your-workspace'
)
```

### 3. Model Catalog

**Available Models:**

**OpenAI Models:**
- GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- DALL-E 3, Whisper
- Embeddings models

**Meta Llama:**
- Llama 3.1 (405B, 70B, 8B)
- Llama 3.2 (90B, 11B, 3B, 1B)
- Vision-enabled models

**Mistral AI:**
- Mistral Large
- Mistral 7B, 8x7B

**Cohere:**
- Command R+, Command R
- Embed models

**Deployment Options:**
- Managed endpoints (Azure-hosted)
- Serverless APIs (pay-per-token)
- Self-hosted (Azure ML)

### 4. Prompt Flow

**What:** Visual designer for LLM applications

**Features:**
- Drag-and-drop workflow builder
- Built-in tools (LLM, Python, Prompt)
- Custom tools
- Debugging and tracing
- Evaluation integration
- CI/CD deployment

**Example Flow:**

```python
from promptflow import tool

@tool
def extract_entities(text: str) -> dict:
    # Custom entity extraction logic
    return {"entities": ["entity1", "entity2"]}

@tool
def llm_call(prompt: str, entities: dict) -> str:
    # Call LLM with context
    return response
```

**Use Cases:**
- RAG applications
- Multi-step reasoning
- Agent workflows
- Data processing pipelines

### 5. AI Foundry Agent Service

**What:** Platform for building enterprise AI agents

**Key Features:**
- Pre-built agent templates
- Multi-agent orchestration
- Tool integration (Microsoft 365, Dynamics, custom APIs)
- Memory and state management
- Conversation management
- Built-in security and compliance

**Agent Types:**

**Conversational Agents:**
- Customer service bots
- Virtual assistants
- FAQ bots

**Task Automation Agents:**
- Workflow automation
- Data processing
- Report generation

**Research Agents:**
- Information gathering
- Document analysis
- Competitive intelligence

**Multi-Agent Systems:**
- Specialized agent teams
- Handoffs between agents
- Coordinated workflows

**Architecture:**

```
User Query → Agent Orchestrator → Specialized Agents
                                   ├─ Research Agent
                                   ├─ Data Agent
                                   └─ Action Agent
                                        ↓
                                   Response
```

### 6. Content Safety

**What:** AI-powered content moderation

**Categories:**
- Hate and fairness
- Sexual content
- Violence
- Self-harm

**Severity Levels:**
- Safe
- Low
- Medium
- High

**Usage:**

```python
from azure.ai.contentsafety import ContentSafetyClient

client = ContentSafetyClient(endpoint, credential)

response = client.analyze_text(
    text="User input text",
    categories=["Hate", "Sexual", "Violence", "SelfHarm"]
)

for category in response.categories_analysis:
    print(f"{category.category}: {category.severity}")
```

### 7. Evaluation and Monitoring

**Evaluation Metrics:**
- Groundedness (factual accuracy)
- Relevance
- Coherence
- Fluency
- Safety and fairness

**Built-in Evaluators:**
- GPT-assisted evaluation
- Human evaluation
- Custom metrics

**Example:**

```python
from azure.ai.evaluation import evaluate

results = evaluate(
    data="test_data.jsonl",
    evaluators={
        "groundedness": groundedness_evaluator,
        "relevance": relevance_evaluator
    }
)

print(results.metrics)
```

## Getting Started

### Prerequisites

1. Azure account
2. Azure AI Foundry hub and project
3. Azure OpenAI resource
4. API keys

### Step 1: Setup

```bash
# Install SDK
pip install azure-ai-ml azure-identity openai

# Set environment variables
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### Step 2: Basic API Usage

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o",  # deployment name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### Step 3: Streaming Response

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### Step 4: Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    # Execute function
    result = get_weather(function_args["location"])
    
    # Send result back
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in Seattle?"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]
    )
```

### Step 5: Building an Agent

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Create project client
project_client = AIProjectClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    project_name="your-project"
)

# Create agent
agent = project_client.agents.create_agent(
    model="gpt-4o",
    name="Customer Service Agent",
    instructions="You are a helpful customer service agent. Be polite and professional.",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"}
    ]
)

# Create thread
thread = project_client.agents.create_thread()

# Add message
message = project_client.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="I need help with my order"
)

# Run agent
run = project_client.agents.create_run(
    thread_id=thread.id,
    agent_id=agent.id
)

# Wait for completion
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = project_client.agents.get_run(thread.id, run.id)

# Get response
messages = project_client.agents.list_messages(thread.id)
print(messages.data[0].content[0].text.value)
```

## Best Practices

### 1. Cost Optimization

**Strategies:**
- Use GPT-3.5 Turbo for simple tasks
- Implement response caching
- Use provisioned throughput for predictable workloads
- Monitor usage with Azure Monitor
- Set spending limits

**Provisioned Throughput:**
- Predictable performance
- Reserved capacity
- Cost savings for high-volume (30-50% at scale)
- Pricing: Per PTU (Provisioned Throughput Unit)

### 2. Security

**Best Practices:**
- Use managed identities (no API keys)
- Private endpoints for VNet isolation
- Customer-managed keys for encryption
- Azure AD authentication
- Role-based access control (RBAC)

**Example:**

```python
from azure.identity import ManagedIdentityCredential

credential = ManagedIdentityCredential()

client = AzureOpenAI(
    azure_ad_token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

### 3. Responsible AI

**Content Filtering:**
- Enabled by default
- Configurable severity levels
- Custom blocklists
- PII detection

**Evaluation:**
- Test for bias and fairness
- Groundedness checks
- Safety evaluations
- Regular audits

### 4. Microsoft 365 Integration

**Copilot Extensibility:**
- Extend Microsoft 365 Copilot
- Custom plugins
- Data connectors
- Enterprise search

**Example Use Cases:**
- Custom business logic in Copilot
- Integration with internal systems
- Specialized knowledge bases
- Workflow automation

## Use Cases

### 1. Enterprise Chatbots
- Customer service automation
- IT helpdesk
- HR assistant
- Sales support

### 2. Document Intelligence
- Contract analysis
- Invoice processing
- Document summarization
- Information extraction

### 3. Code Assistance
- Code generation
- Code review
- Documentation generation
- Bug detection

### 4. Multi-Agent Systems
- Research and analysis teams
- Customer journey automation
- Complex workflow orchestration
- Specialized task handling

## Comparison: Azure AI Foundry vs. Competitors

| Feature | Azure AI Foundry | AWS Bedrock | Google Vertex AI |
|---------|-----------------|-------------|------------------|
| **Primary Models** | OpenAI (GPT-4) | Claude, Llama | Gemini |
| **Max Context** | 128K | 200K | 2M |
| **Agent Service** | AI Foundry Agent Service | Bedrock Agents | Custom |
| **M365 Integration** | Native | Via API | Via API |
| **Enterprise Focus** | Strong | Strong | Moderate |
| **Pricing** | Similar to OpenAI | Varies | Competitive |
| **Best For** | Microsoft ecosystem | AWS ecosystem | Google ecosystem |

## Limitations

**Current Limitations:**
- OpenAI model availability varies by region
- Rate limits (vary by deployment)
- Some features in preview
- Requires Azure subscription

**Considerations:**
- Learning curve for Azure services
- Pricing can be complex
- Regional availability
- Quota management

## Resources

- Azure AI Foundry Documentation - https://learn.microsoft.com/azure/ai-studio/
- Azure OpenAI Documentation - https://learn.microsoft.com/azure/ai-services/openai/
- Azure AI Foundry Pricing - https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
- Microsoft Learn - https://learn.microsoft.com/training/
- Azure AI Samples - https://github.com/Azure-Samples/azureai-samples

## Next Steps

- **[OpenAI Agent Builder](./06-openai-agent-builder.md)** - OpenAI's agent framework
- **[Cost Management](./07-cost-management.md)** - Optimize cloud spending
- **[Cloud Services Overview](./CLOUD_SERVICES_OVERVIEW_2025.md)** - Compare all services


