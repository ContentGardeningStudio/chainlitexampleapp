# Chainlit LangChain with RAG example - Chat with documents

## Installation

### Python requirements

In your Python 3.13 virtual env, install all deps by running:

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Provide the OpenAI API key using the OPENAI_API_KEY env var.

```bash
chainlit run app.py --port 8000
```

## To do list

1. Optionally use Ollama
  - Ollama model for the chat
  - Ollama embedding model

2. Upgrade to Chainlit 2.5.5
   See https://docs.chainlit.io/guides/migration/2.0.0

