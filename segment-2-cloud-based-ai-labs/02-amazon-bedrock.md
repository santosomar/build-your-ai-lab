# Amazon Bedrock

## Introduction

Amazon Bedrock is AWS's fully managed service that provides access to high-performing foundation models (FMs) from leading AI companies through a single API. It enables you to build and scale generative AI applications without managing infrastructure.

## What is Amazon Bedrock?

**Amazon Bedrock** is a serverless service that offers:
- Access to multiple foundation models via unified API
- Customization capabilities (fine-tuning, RAG)
- Enterprise-grade security and privacy
- Pay-per-use pricing model
- No infrastructure management required

**Key Concept:**
Instead of deploying and managing your own LLMs, Bedrock provides API access to pre-trained models from various providers, similar to OpenAI's API but with more model choices and AWS integration.

## Available Foundation Models (2025)

### Amazon Titan Models

**Amazon Titan Text:**
- Titan Text Express - Fast, cost-effective text generation
- Titan Text Lite - Lightweight for simple tasks
- Titan Text Premier - Advanced reasoning and long context

**Amazon Titan Embeddings:**
- Titan Embeddings G1 - Text - 8K context, 1,536 dimensions
- Titan Multimodal Embeddings - Text and images

**Amazon Titan Image Generator:**
- Generate, edit, and customize images
- Inpainting and outpainting
- Background removal
- Subject-guided generation

**Pricing (Approximate):**
- Titan Text Express: $0.0002/1K input tokens, $0.0006/1K output tokens
- Titan Embeddings: $0.0001/1K tokens
- Titan Image: $0.008/image

### Anthropic Claude Models

**Claude 3.5 Sonnet (Latest):**
- 200K token context window
- Superior reasoning and coding
- Vision capabilities
- Best for: Complex analysis, coding, research

**Claude 3 Opus:**
- Highest capability model
- Best for: Complex tasks requiring deep understanding

**Claude 3 Sonnet:**
- Balanced performance and speed
- Best for: General-purpose applications

**Claude 3 Haiku:**
- Fastest, most compact
- Best for: High-volume, simple tasks

**Pricing (Approximate):**
- Claude 3.5 Sonnet: $0.003/1K input, $0.015/1K output
- Claude 3 Opus: $0.015/1K input, $0.075/1K output
- Claude 3 Haiku: $0.00025/1K input, $0.00125/1K output

### Meta Llama Models

**Llama 3.1 (405B, 70B, 8B):**
- Open weights model available via Bedrock
- Multilingual support (8 languages)
- 128K context window
- Best for: General-purpose, code generation

**Llama 3.2 (90B, 11B, 3B, 1B):**
- Vision-enabled models (11B, 90B)
- Lightweight models (1B, 3B) for edge
- Multilingual
- Best for: Multimodal tasks, edge deployment

**Pricing (Approximate):**
- Llama 3.1 405B: $0.00532/1K input, $0.016/1K output
- Llama 3.1 70B: $0.00099/1K input, $0.00099/1K output
- Llama 3.1 8B: $0.0003/1K input, $0.0006/1K output

### Cohere Models

**Cohere Command R+:**
- Optimized for RAG and tool use
- Multilingual (10 languages)
- 128K context window
- Best for: Enterprise search, chatbots

**Cohere Command R:**
- Balanced performance
- Cost-effective
- Best for: General conversational AI

**Cohere Embed:**
- Multilingual embeddings
- Best for: Semantic search, clustering

**Pricing (Approximate):**
- Command R+: $0.003/1K input, $0.015/1K output
- Command R: $0.0005/1K input, $0.0015/1K output
- Embed: $0.0001/1K tokens

### AI21 Labs Jurassic-2

**Jurassic-2 Ultra:**
- Advanced language understanding
- Best for: Complex text generation

**Jurassic-2 Mid:**
- Balanced performance
- Best for: General applications

**Pricing (Approximate):**
- Jurassic-2 Ultra: $0.0188/1K tokens
- Jurassic-2 Mid: $0.0125/1K tokens

### Stability AI

**Stable Diffusion XL:**
- High-quality image generation
- 1024x1024 resolution
- Style presets available

**Pricing (Approximate):**
- $0.04 per image (1024x1024)

### Mistral AI

**Mistral Large:**
- 128K context window
- Multilingual (English, French, German, Spanish, Italian)
- Function calling support
- Best for: Complex reasoning, multilingual tasks

**Mistral 7B/8x7B:**
- Open weights models
- Cost-effective
- Best for: General-purpose applications

**Pricing (Approximate):**
- Mistral Large: $0.008/1K input, $0.024/1K output
- Mistral 8x7B: $0.00045/1K input, $0.0007/1K output

## Key Features

### 1. Model Customization

**Fine-Tuning:**
- Customize models with your data
- Continued pre-training
- Adapter-based fine-tuning (LoRA)
- Private, isolated training

**Process:**
```
1. Prepare training data (JSONL format)
2. Upload to S3
3. Create fine-tuning job via console or API
4. Deploy custom model
5. Use via same Bedrock API
```

**Use Cases:**
- Domain-specific language (legal, medical)
- Brand voice and tone
- Specialized knowledge
- Task-specific optimization

**Pricing:**
- Training: $0.008-$0.024 per 1K tokens (varies by base model)
- Storage: $1.95 per month per model
- Inference: Base model price + 20-50% premium

### 2. Retrieval-Augmented Generation (RAG)

**Knowledge Bases for Amazon Bedrock:**
- Connect models to your data sources
- Automatic chunking and embedding
- Vector storage (OpenSearch Serverless, Pinecone, Redis)
- Semantic search and retrieval

**Architecture:**
```
User Query → Bedrock → Knowledge Base → Vector DB
                ↓
         Retrieve Context
                ↓
         Generate Response
```

**Supported Data Sources:**
- Amazon S3
- Web crawlers
- Confluence
- Salesforce
- SharePoint
- Custom connectors

**Benefits:**
- Reduce hallucinations
- Keep models up-to-date without retraining
- Source attribution
- Domain-specific knowledge

**Pricing:**
- Storage: OpenSearch Serverless costs
- Embeddings: $0.0001-$0.0002/1K tokens
- Retrieval: Included in inference cost

### 3. Agents for Amazon Bedrock

**What are Agents?**
Autonomous AI systems that can:
- Break down complex tasks
- Make API calls
- Use tools and functions
- Maintain conversation context
- Execute multi-step workflows

**Components:**
- **Foundation Model:** Brain of the agent
- **Instructions:** Define agent behavior
- **Action Groups:** APIs/functions agent can call
- **Knowledge Bases:** Information sources

**Example Use Cases:**
- Customer service automation
- Data analysis and reporting
- Workflow automation
- Research assistants
- Code generation and debugging

**Pricing:**
- Base model inference costs
- API call costs (your APIs)
- Knowledge base costs (if used)

### 4. Guardrails for Amazon Bedrock

**Content Filtering:**
- Denied topics
- Content filters (hate, violence, sexual, etc.)
- Word filters (profanity, custom blocklists)
- Sensitive information filters (PII, financial data)

**Safety Levels:**
- None
- Low
- Medium
- High

**Application:**
- Input filtering (user prompts)
- Output filtering (model responses)
- Both directions

**Use Cases:**
- Child-safe applications
- Enterprise compliance
- Brand safety
- Regulatory requirements

**Pricing:**
- $0.75 per 1,000 content units (input)
- $1.00 per 1,000 content units (output)

### 5. Model Evaluation

**Automatic Evaluation:**
- Compare model outputs
- Benchmark against ground truth
- Custom evaluation metrics
- A/B testing

**Metrics:**
- Accuracy
- Robustness
- Toxicity
- Factuality

**Human Evaluation:**
- SageMaker Ground Truth integration
- Custom evaluation workflows
- Quality assessment

## Getting Started

### Prerequisites

1. **AWS Account**
2. **IAM Permissions**
3. **Model Access** (request via console)

### Step 1: Request Model Access

```bash
# Via AWS Console:
1. Navigate to Amazon Bedrock
2. Click "Model access"
3. Select models you want
4. Click "Request model access"
5. Wait for approval (usually instant for most models)
```

### Step 2: Basic API Usage

**Python SDK (boto3):**

```python
import boto3
import json

# Create Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# Prepare request
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms"
        }
    ]
})

# Invoke model
response = bedrock.invoke_model(
    modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=body
)

# Parse response
response_body = json.loads(response['body'].read())
print(response_body['content'][0]['text'])
```

**Streaming Response:**

```python
response = bedrock.invoke_model_with_response_stream(
    modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=body
)

# Process stream
stream = response['body']
for event in stream:
    chunk = json.loads(event['chunk']['bytes'])
    if chunk['type'] == 'content_block_delta':
        print(chunk['delta']['text'], end='', flush=True)
```

### Step 3: Using LangChain

```python
from langchain_aws import ChatBedrock

# Initialize
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1"
)

# Simple query
response = llm.invoke("What is machine learning?")
print(response.content)

# With system prompt
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in {domain}."),
    ("human", "{question}")
])

chain = template | llm
response = chain.invoke({
    "domain": "data science",
    "question": "Explain gradient descent"
})
print(response.content)
```

### Step 4: Creating a Knowledge Base

```python
import boto3

bedrock_agent = boto3.client('bedrock-agent')

# Create knowledge base
kb_response = bedrock_agent.create_knowledge_base(
    name='my-knowledge-base',
    description='Company documentation',
    roleArn='arn:aws:iam::123456789012:role/BedrockKBRole',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1'
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': 'arn:aws:aoss:us-east-1:123456789012:collection/abc123',
            'vectorIndexName': 'bedrock-knowledge-base',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
)

# Add data source (S3)
bedrock_agent.create_data_source(
    knowledgeBaseId=kb_response['knowledgeBase']['knowledgeBaseId'],
    name='s3-docs',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': 'arn:aws:s3:::my-docs-bucket'
        }
    }
)
```

## Best Practices

### 1. Model Selection

**Choose Based On:**
- **Task Complexity:** Haiku for simple, Opus for complex
- **Cost:** Balance performance vs. price
- **Latency:** Smaller models are faster
- **Context Length:** Match your needs
- **Capabilities:** Vision, function calling, etc.

**Decision Matrix:**
| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Simple Q&A | Claude Haiku, Llama 8B | Fast, cost-effective |
| Complex Analysis | Claude 3.5 Sonnet, Opus | Superior reasoning |
| Code Generation | Claude 3.5 Sonnet, Llama 70B | Strong coding ability |
| RAG Applications | Command R+, Claude Sonnet | Optimized for retrieval |
| High Volume | Titan Express, Llama 8B | Low cost per request |
| Multimodal | Claude 3.5, Llama 3.2 90B | Vision support |

### 2. Cost Optimization

**Strategies:**
- Use smaller models when possible
- Implement caching for repeated queries
- Batch requests when feasible
- Use streaming for better UX (doesn't reduce cost)
- Monitor usage with CloudWatch
- Set billing alerts

**Example Savings:**
```
Scenario: 1M tokens/month

Claude Opus: $15/1K input = $15,000
Claude Sonnet: $3/1K input = $3,000
Claude Haiku: $0.25/1K input = $250

Savings: $14,750 by choosing appropriate model
```

### 3. Prompt Engineering

**Best Practices:**
- Be specific and clear
- Provide examples (few-shot learning)
- Use system prompts for consistent behavior
- Structure complex prompts with XML tags
- Test and iterate

**Example:**
```python
prompt = """
<task>
Analyze the following customer review and extract:
1. Sentiment (positive/negative/neutral)
2. Key topics mentioned
3. Actionable feedback
</task>

<review>
{review_text}
</review>

<output_format>
Return as JSON with keys: sentiment, topics, feedback
</output_format>
"""
```

### 4. Security and Compliance

**Data Protection:**
- Data not used to train base models
- Encryption in transit and at rest
- VPC endpoints for private connectivity
- IAM for access control
- CloudTrail for audit logging

**Compliance:**
- HIPAA eligible
- SOC 1, 2, 3
- ISO 27001
- PCI DSS
- GDPR compliant

### 5. Error Handling

```python
import boto3
from botocore.exceptions import ClientError
import time

def invoke_with_retry(bedrock, model_id, body, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=body
            )
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'ThrottlingException':
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Throttled. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif error_code == 'ModelTimeoutException':
                print("Model timeout. Retrying...")
                continue
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Use Cases and Examples

### 1. Content Generation

```python
# Blog post generation
prompt = """
Write a 500-word blog post about the benefits of cloud computing for small businesses.
Include:
- Introduction
- 3 main benefits
- Conclusion
- SEO-friendly
"""

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}]
    })
)
```

### 2. Document Summarization

```python
# Long document summarization
prompt = f"""
Summarize the following document in 3 bullet points:

<document>
{long_document}
</document>

Focus on the key takeaways and actionable insights.
"""
```

### 3. Code Generation

```python
# Python function generation
prompt = """
Write a Python function that:
1. Takes a list of dictionaries
2. Filters by a given key-value pair
3. Sorts by another key
4. Returns top N results

Include docstring and type hints.
"""
```

### 4. Data Analysis

```python
# CSV analysis
prompt = f"""
Analyze this sales data and provide insights:

<data>
{csv_data}
</data>

Provide:
1. Summary statistics
2. Trends
3. Recommendations
"""
```

## Monitoring and Observability

**CloudWatch Metrics:**
- Invocations
- Latency (P50, P90, P99)
- Errors
- Throttles
- Token usage

**Logging:**
- Model invocations
- Input/output (optional)
- Errors and exceptions
- Custom metrics

**Cost Tracking:**
- AWS Cost Explorer
- Tag resources for cost allocation
- Set up billing alerts
- Use AWS Budgets

## Limitations and Considerations

**Current Limitations:**
- Model availability varies by region
- Some models require approval
- Rate limits (vary by model and account)
- Context window limits (model-dependent)
- No fine-tuning for all models

**Best Practices:**
- Test in multiple regions for availability
- Request limit increases proactively
- Implement retry logic
- Monitor usage patterns
- Plan for model updates/deprecations

## Comparison with Alternatives

| Feature | Bedrock | OpenAI API | Azure OpenAI | Vertex AI |
|---------|---------|------------|--------------|-----------|
| **Model Choice** | Multiple providers | OpenAI only | OpenAI only | Google + others |
| **Customization** | Fine-tuning, RAG | Fine-tuning | Fine-tuning | Fine-tuning, RAG |
| **Pricing** | Varies by model | Fixed per model | Similar to OpenAI | Varies |
| **AWS Integration** | Native | Via SDK | Via SDK | Via SDK |
| **Data Privacy** | Not used for training | Not used (paid) | Not used | Not used |
| **Deployment** | Serverless | API | API | API + Vertex |


## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Bedrock Workshop](https://catalog.workshops.aws/bedrock/)
- [LangChain Bedrock](https://python.langchain.com/docs/integrations/llms/bedrock)
- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
- [AWS Skill Builder](https://skillbuilder.aws/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)

## Next Steps

- **[Amazon SageMaker](./03-amazon-sagemaker.md)** - Full ML platform
- **[Google Vertex AI](./04-google-vertex-ai.md)** - Google's AI platform
- **[Azure AI Foundry](./05-azure-ai-foundry.md)** - Microsoft's AI tools
- **[Cost Management](./07-cost-management.md)** - Optimize cloud spending


