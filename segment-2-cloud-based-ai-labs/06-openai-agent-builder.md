# OpenAI Agent Builder and Agent Kit

## Introduction

OpenAI provides powerful tools for building AI agents through its Assistants API and experimental frameworks like Swarm. These tools enable developers to create autonomous agents that can use tools, maintain conversation context, and coordinate with other agents to accomplish complex tasks.

## What are OpenAI Agents?

**OpenAI Agents** are AI systems that can:
- Maintain conversation context across multiple turns
- Use tools and functions to take actions
- Access and search files
- Execute code
- Make decisions autonomously
- Coordinate with other agents

**Key Components:**
- **Assistants API:** Production-ready agent framework
- **Swarm:** Experimental multi-agent orchestration
- **Function Calling:** Tool use capabilities
- **Code Interpreter:** Execute Python code
- **File Search:** RAG capabilities

## Assistants API

### What is the Assistants API?

The Assistants API allows you to build AI assistants that can:
- Have persistent instructions and personality
- Access multiple tools simultaneously
- Maintain conversation threads
- Store and retrieve files
- Execute code safely

**Architecture:**

```
Assistant (persistent)
  ├─ Instructions (system prompt)
  ├─ Model (gpt-4, gpt-4o, etc.)
  ├─ Tools (functions, code_interpreter, file_search)
  └─ Files (documents for context)

Thread (conversation)
  ├─ Messages (user and assistant)
  └─ Runs (executions)
```

### Core Concepts

**1. Assistants:**
- Persistent AI entities
- Configured with instructions, model, and tools
- Can be reused across multiple conversations

**2. Threads:**
- Conversation sessions
- Store message history
- Automatically manage context window

**3. Messages:**
- User and assistant messages
- Can include text and files
- Stored in threads

**4. Runs:**
- Execution of assistant on a thread
- Can require action (function calls)
- Async processing

### Getting Started

**Installation:**

```bash
pip install openai
```

**Basic Usage:**

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Create assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a helpful math tutor. Explain concepts clearly.",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)

# Create thread
thread = client.beta.threads.create()

# Add message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve the equation: 3x + 11 = 14"
)

# Run assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for completion
import time
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

# Get response
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
```

### Available Tools

**1. Code Interpreter:**
- Execute Python code
- Generate visualizations
- Process data files
- Perform calculations

**Example:**

```python
assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="Analyze data and create visualizations",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)

# Upload file
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose="assistants"
)

# Create thread with file
thread = client.beta.threads.create(
    messages=[{
        "role": "user",
        "content": "Analyze this data and create a chart",
        "attachments": [{"file_id": file.id, "tools": [{"type": "code_interpreter"}]}]
    }]
)
```

**2. File Search (RAG):**
- Search through uploaded documents
- Retrieve relevant information
- Answer questions based on files

**Example:**

```python
# Create vector store
vector_store = client.beta.vector_stores.create(
    name="Product Documentation"
)

# Upload files
file_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
file_streams = [open(path, "rb") for path in file_paths]

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams
)

# Create assistant with file search
assistant = client.beta.assistants.create(
    name="Documentation Assistant",
    instructions="Answer questions based on the documentation",
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
)
```

**3. Function Calling:**
- Call external APIs
- Execute custom logic
- Integrate with systems

**Example:**

```python
import json

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

assistant = client.beta.assistants.create(
    name="Weather Assistant",
    instructions="Help users check the weather",
    model="gpt-4o",
    tools=tools
)

# Run with function handling
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Check for required action
while run.status == "requires_action":
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    
    tool_outputs = []
    for tool_call in tool_calls:
        if tool_call.function.name == "get_weather":
            args = json.loads(tool_call.function.arguments)
            result = get_weather(args["location"], args.get("unit", "celsius"))
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            })
    
    # Submit outputs
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )
    
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
```

### Streaming

```python
from openai import AssistantEventHandler

class EventHandler(AssistantEventHandler):
    def on_text_created(self, text):
        print("\nassistant > ", end="", flush=True)
    
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
    
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

# Stream response
with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    event_handler=EventHandler()
) as stream:
    stream.until_done()
```

## OpenAI Swarm (Experimental)

### What is Swarm?

**Swarm** is an experimental framework for building multi-agent systems with:
- Lightweight agent coordination
- Handoffs between agents
- Context management
- Routine-based execution

**Status:** Experimental (not for production)

**Key Concepts:**

**1. Agents:**
- Specialized for specific tasks
- Have instructions and functions
- Can hand off to other agents

**2. Handoffs:**
- Transfer conversation to another agent
- Pass context
- Coordinated workflows

**3. Routines:**
- Sequences of agent actions
- Conditional logic
- Multi-step processes

### Installation

```bash
pip install git+https://github.com/openai/swarm.git
```

### Basic Usage

```python
from swarm import Swarm, Agent

client = Swarm()

# Define agents
def transfer_to_sales():
    """Transfer to sales agent"""
    return sales_agent

support_agent = Agent(
    name="Support Agent",
    instructions="You handle customer support inquiries. Transfer to sales for purchases.",
    functions=[transfer_to_sales]
)

sales_agent = Agent(
    name="Sales Agent",
    instructions="You help customers make purchases."
)

# Run conversation
messages = [{"role": "user", "content": "I want to buy a product"}]

response = client.run(
    agent=support_agent,
    messages=messages
)

print(response.messages[-1]["content"])
```

### Multi-Agent Example

```python
from swarm import Swarm, Agent

client = Swarm()

# Research agent
def search_web(query: str) -> str:
    """Search the web for information"""
    # Implementation
    return f"Search results for: {query}"

research_agent = Agent(
    name="Researcher",
    instructions="You research topics and gather information",
    functions=[search_web, lambda: writer_agent]  # Can transfer to writer
)

# Writer agent
def save_document(content: str) -> str:
    """Save document to file"""
    # Implementation
    return "Document saved"

writer_agent = Agent(
    name="Writer",
    instructions="You write articles based on research",
    functions=[save_document]
)

# Orchestrate
messages = [{"role": "user", "content": "Research AI trends and write an article"}]

response = client.run(
    agent=research_agent,
    messages=messages
)
```

### Swarm Patterns

**1. Triage Pattern:**
```python
# Router agent decides which specialist to use
triage_agent = Agent(
    name="Triage",
    instructions="Route users to the right specialist",
    functions=[transfer_to_tech, transfer_to_billing, transfer_to_sales]
)
```

**2. Sequential Pattern:**
```python
# Agents work in sequence
step1_agent = Agent(functions=[lambda: step2_agent])
step2_agent = Agent(functions=[lambda: step3_agent])
step3_agent = Agent()
```

**3. Parallel Pattern:**
```python
# Multiple agents work simultaneously
def run_parallel():
    results = []
    for agent in [agent1, agent2, agent3]:
        response = client.run(agent=agent, messages=messages)
        results.append(response)
    return results
```

## Best Practices

### 1. Assistant Design

**Instructions:**
- Be specific and clear
- Include examples
- Define personality and tone
- Specify output format

**Example:**
```python
instructions = """
You are a professional customer service agent for TechCorp.

Guidelines:
- Always be polite and professional
- Gather customer information before troubleshooting
- Offer solutions step-by-step
- Escalate to human if unable to resolve

Response Format:
1. Acknowledge the issue
2. Ask clarifying questions
3. Provide solution
4. Confirm resolution
"""
```

### 2. Tool Selection

**Choose Tools Based On:**
- **Code Interpreter:** Data analysis, calculations, visualizations
- **File Search:** Document Q&A, knowledge bases
- **Functions:** External integrations, actions

**Combine Tools:**
```python
tools = [
    {"type": "code_interpreter"},
    {"type": "file_search"},
    {"type": "function", "function": custom_function}
]
```

### 3. Error Handling

```python
try:
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Poll with timeout
    timeout = 60
    start_time = time.time()
    
    while run.status in ["queued", "in_progress"]:
        if time.time() - start_time > timeout:
            client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
            raise TimeoutError("Run exceeded timeout")
        
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
    
except Exception as e:
    print(f"Error: {e}")
```

### 4. Cost Management

**Strategies:**
- Use GPT-3.5 Turbo for simple tasks
- Limit file sizes for file search
- Set max_prompt_tokens and max_completion_tokens
- Monitor usage via API
- Clean up old threads and files

**Pricing:**
- Base model costs (per token)
- Code Interpreter: $0.03 per session
- File Search: $0.10 per GB/day (vector store)
- Storage: $0.10 per GB/day (files)

## Use Cases

### 1. Customer Support
- Multi-turn conversations
- Access to knowledge base
- Ticket creation
- Escalation handling

### 2. Personal Assistants
- Task management
- Calendar integration
- Email drafting
- Research assistance

### 3. Code Assistants
- Code generation
- Debugging
- Documentation
- Code review

### 4. Data Analysis
- Process CSV/Excel files
- Generate visualizations
- Statistical analysis
- Report generation

### 5. Multi-Agent Systems
- Research and writing teams
- Customer journey automation
- Complex workflow orchestration
- Specialized task handling

## Comparison: Assistants API vs. Alternatives

| Feature | OpenAI Assistants | Azure AI Foundry Agents | Bedrock Agents | LangChain |
|---------|------------------|------------------------|----------------|-----------|
| **Ease of Use** | High | Medium | Medium | Low |
| **Multi-Agent** | Swarm (experimental) | Built-in | Limited | LangGraph |
| **File Search** | Built-in | Built-in | Knowledge Bases | Custom |
| **Code Execution** | Built-in | Limited | No | Custom |
| **Pricing** | Per-token + tools | Per-token | Per-token | Free (framework) |
| **Production Ready** | Yes | Yes | Yes | Yes |
| **Best For** | Quick start | Microsoft ecosystem | AWS ecosystem | Full control |

## Limitations

**Current Limitations:**
- Swarm is experimental (not for production)
- Rate limits apply
- File size limits (512MB per file, 10GB per assistant)
- Code Interpreter limited to Python
- No direct internet access

**Considerations:**
- Cost can accumulate with file storage
- Thread management required
- Function calling requires polling
- Limited control over RAG process

## Resources

- [Assistants API Documentation](https://platform.openai.com/docs/assistants/overview)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Swarm GitHub](https://github.com/openai/swarm)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [OpenAI Community](https://community.openai.com/)

## Next Steps

- **[Cost Management](./07-cost-management.md)** - Optimize cloud AI spending
- **[Cloud Services Overview](./CLOUD_SERVICES_OVERVIEW_2025.md)** - Compare all services
- **[Segment 3](../segment-3-integrating-and-leveraging-ai-environments/)** - Hybrid approaches



