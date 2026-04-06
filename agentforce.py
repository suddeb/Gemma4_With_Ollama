"""
Multi-turn Salesforce/Agentforce domain chatbot powered by Gemma 4 via Ollama.

Initialises a conversation with a Salesforce Solution Architect system prompt,
fires a hard-coded seed question (Turn 1), then hands control to an interactive
REPL loop that maintains the full message history across turns so the model
retains context.

Usage::

    python agentforce.py

Type ``quit`` or ``exit`` at the prompt to end the session.

Requirements:
    - Ollama daemon running locally (``ollama serve``)
    - ``gemma4:e2b`` model pulled (``ollama pull gemma4:e2b``)
    - ``ollama`` Python package installed in the active virtual environment
"""

import ollama
import textwrap

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

# System prompt — native system role support in Gemma 4!
messages = [{
    'role': 'system',
    'content': textwrap.dedent('''\
        You are an expert Salesforce Solution Architect.
        You specialize in Agentforce, Data Cloud, and Einstein AI.
        When given a business requirement, reason step-by-step
        before proposing a technical solution.''')
}]
"""list[dict]: Accumulated message history sent to the model on every call.

Each element is a ``{'role': ..., 'content': ...}`` dict following the
OpenAI-compatible chat format that Ollama accepts.  The list grows with
every user/assistant exchange so the model has full conversational context.
"""

# ---------------------------------------------------------------------------
# Turn 1 — seed question
# ---------------------------------------------------------------------------

messages.append({'role': 'user', 'content':
    'We need an SDR AI agent that qualifies inbound leads, '
    'scores them using Data Cloud, and routes to the right rep.'})
print("User: We need an SDR AI agent that qualifies inbound leads...\n")

response = ollama.chat(model='gemma4:e2b', messages=messages)

# Safely extract content depending on ollama-python version (v0.2.0+ uses objects)
ai_content = response.message.content if hasattr(response, 'message') else response['message']['content']
print(f"AI: {ai_content}\n")

# Add response to history for next turn
messages.append({'role': 'assistant', 'content': ai_content})

# ---------------------------------------------------------------------------
# Turn 2+ — interactive loop
# ---------------------------------------------------------------------------

def chat_loop(messages: list[dict]) -> None:
    """Run an interactive multi-turn chat session against the Gemma 4 model.

    Reads user input from stdin, appends it to *messages*, calls
    ``ollama.chat``, prints the model reply, and appends the assistant
    response back to *messages*.  The loop exits when the user types
    ``quit`` or ``exit`` (case-insensitive).

    The function mutates *messages* in-place so the caller retains the
    full conversation history after the loop ends.

    :param messages: Existing conversation history including the system
        prompt and any prior turns.  Each element must be a dict with
        ``'role'`` and ``'content'`` keys.
    :type messages: list[dict]

    Example::

        history = [{'role': 'system', 'content': 'You are helpful.'}]
        chat_loop(history)
        # >>> User (type 'quit' to exit): tell me about Agentforce
        # >>> AI: Agentforce is ...
    """
    while True:
        user_input = input("User (type 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break

        messages.append({'role': 'user', 'content': user_input})
        response = ollama.chat(model='gemma4:e2b', messages=messages)

        ai_content = response.message.content if hasattr(response, 'message') else response['message']['content']
        print(f"\nAI: {ai_content}\n")
        messages.append({'role': 'assistant', 'content': ai_content})


chat_loop(messages)
