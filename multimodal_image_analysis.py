"""
Single-shot multimodal image analysis using Gemma 4 via Ollama.

Reads ``diagram.png`` from the current working directory, sends it alongside
a plain-English prompt to the ``gemma4:e2b`` model, and prints the model's
description to stdout.

Usage::

    python multimodal_image_analysis.py

Requirements:
    - Ollama daemon running locally (``ollama serve``)
    - ``gemma4:e2b`` model pulled (``ollama pull gemma4:e2b``)
    - ``diagram.png`` present in the current working directory
    - ``ollama`` Python package installed in the active virtual environment

Note:
    The ``images`` field in the message dict accepts a list of file paths
    (strings).  Ollama encodes each image as base-64 before sending it to
    the model.  This script uses the dict-access style
    (``response['message']['content']``) which is compatible with older
    ollama-python releases; newer releases (v0.2.0+) also expose
    ``response.message.content`` as an object attribute.
"""

import ollama

response = ollama.chat(
    model='gemma4:e2b',
    messages=[{
        'role': 'user',
        'content': 'Analyze this diagram and explain what it shows in plain English.',
        'images': ['diagram.png']
    }]
)

print(response['message']['content'])
