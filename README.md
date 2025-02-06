# GenAI Agents System

This project demonstrates a multi-agent system combining Google Gemini, OpenAI, and automated maintenance capabilities:

**Core Components:**
- `Google.py`: Google Gemini integration for advanced AI capabilities
- `OpenAI.py`: OpenAI integration with parallel tool execution
- `MaintainerAgent.py`: AI-powered code maintenance with MLOps features

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set API keys:
```bash
export GOOGLE_API_KEY="your_google_key"
export OPENAI_API_KEY="your_openai_key"
```

Get keys from:
- [Google AI Studio](https://makersuite.google.com/app/apikey)
- [OpenAI Platform](https://platform.openai.com/api-keys)

## Features

### Google AI Core (`Google.py`)
- Text embeddings & similarity analysis
- RAG document Q&A system
- Model fine-tuning and version management
- LangGraph agent workflows
- Batch document processing with caching

### Maintainer Agent (`MaintainerAgent.py`)
- Automated deprecation checks
- Security compliance scanning
- Model performance monitoring
- Automated GitHub issue creation
- Scheduled maintenance cycles
- Test coverage validation

### OpenAI Integration (`OpenAI.py`)
- Parallel function calling
- Neural classification models
- Stateful agent conversations
- Embedding-based search
- Custom tool definitions

## Usage

Run the maintenance agent:
```bash
python MaintainerAgent.py
```

Or test individual components:
```bash
# Google features
python Google.py --all

# OpenAI features 
python OpenAI.py --rag --classify
```

**Key Workflows:**
1. Document processing pipeline (Add -> Embed -> Query)
2. Model fine-tuning and evaluation
3. Automated code maintenance checks
4. Cross-provider function calling
5. Performance monitoring and retraining

## Example Output

```
=== Maintenance Check ===
Detected 2 deprecated API usages
Created issue #42: Update Gemini embedding model
Verified test coverage (82% coverage)

=== Document Search ===
Query: "Neural network applications"
Top match: "Deep learning fundamentals" (0.92 similarity)

=== Model Performance ===
Current accuracy: 94.2% | Latency: 1.2s
```

## Customization

1. Add new maintenance rules in `MaintainerAgent.py`
2. Extend Google/OpenAI agents with custom tools
3. Modify model architectures in `Google.py`/`OpenAI.py`
4. Add new documentation sources
5. Configure performance thresholds

## License
Apache License 2.0 - See LICENSE for details.