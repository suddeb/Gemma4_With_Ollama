import ollama
import textwrap

# System prompt — native system role support in Gemma 4!
messages = [{
    'role': 'system',
    'content': textwrap.dedent('''\
        You are an expert Salesforce Solution Architect.
        You specialize in Agentforce, Data Cloud, and Einstein AI.
        When given a business requirement, reason step-by-step
        before proposing a technical solution.''')
}]

# Turn 1
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

# Turn 2+ Interactive Loop
while True:
    user_input = input("User (type 'quit' to exit): ")
    if user_input.lower() in ['quit', 'exit']:
        break
        
    messages.append({'role': 'user', 'content': user_input})
    response = ollama.chat(model='gemma4:e2b', messages=messages)
    
    ai_content = response.message.content if hasattr(response, 'message') else response['message']['content']
    print(f"\nAI: {ai_content}\n")
    messages.append({'role': 'assistant', 'content': ai_content})