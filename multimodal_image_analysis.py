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
