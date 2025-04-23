# Chainlit LangChain with RAG demo - Chat with documents

## Installation

### Python requirements

In your Python virtual env (3.12 and 3.13 tested), install all deps by running:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Provide the OpenAI API key using the OPENAI_API_KEY env var.

```bash
chainlit run app.py --port 8000
```

## To do list

1. Optionally use an Ollama model for the chat
2. Optionally use an Ollama embedding model
