# Gemma 4 with Ollama (Local AI1)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Experiments with **Gemma 4** running locally via [Ollama](https://ollama.com). All scripts use the `gemma4:e2b` model. There is also a standalone FastAPI/SQLite items API (`main.py`) unrelated to the Ollama experiments.

## Environment Setup

The project uses a `.venv` with Python 3.14. Activate it before running any scripts:

```bash
source .venv/bin/activate
```

### Python Dependencies

Install all required packages at once using the bundled requirements file:

```bash
pip install -r requirements.txt
```

| Package | Version | Used by |
|---------|---------|---------|
| `ollama` | latest | `agentforce.py`, `multimodal_image_analysis.py` |
| `fastapi` | latest | `FastApi_Crud_Server.py` |
| `pydantic` | latest | `FastApi_Crud_Server.py` |
| `sqlalchemy` | latest | `FastApi_Crud_Server.py` |
| `uvicorn` | latest | `FastApi_Crud_Server.py` (ASGI server) |

> `textwrap` is part of the Python standard library — no install needed.

### Ollama Setup

Ollama must be running locally with the `gemma4:e2b` model pulled:

```bash
ollama serve          # start Ollama daemon (if not running)
ollama pull gemma4:e2b
```

## Running the Scripts

```bash
# Multi-turn Salesforce/Agentforce domain chatbot
python agentforce.py

# Multimodal image analysis (reads diagram.png)
python multimodal_image_analysis.py

# FastAPI items CRUD server (SQLite backend)
python FastApi_Crud_Server.py
# or: uvicorn FastApi_Crud_Server:app --reload
```

The FastAPI server runs on `http://0.0.0.0:8000`. The single endpoint is `POST /items` accepting `{name, price, in_stock}` and persisting to `items.db`.

## Architecture

| File | Purpose |
|------|---------|
| `agentforce.py` | Multi-turn chat loop using `ollama.chat()` with a Salesforce Solution Architect system prompt; maintains message history across turns |
| `multimodal_image_analysis.py` | Single-shot multimodal call passing `diagram.png` as an image alongside a text prompt |
| `FastApi_Crud_Server.py` | FastAPI app with Pydantic input validation → SQLAlchemy ORM → SQLite (`items.db`); no dependency injection for the DB session (uses a manual open/close pattern) |

### Ollama API pattern

Both `agentforce.py` and `multimodal_image_analysis.py` use the `ollama` Python package. The response shape differs between ollama-python versions — `agentforce.py` handles both with:
```python
ai_content = response.message.content if hasattr(response, 'message') else response['message']['content']
```
