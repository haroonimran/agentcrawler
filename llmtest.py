import ollama

def hello_world_agent(query):
    # Create a client
    client = ollama.Client()
    
    # Generate a response
    response = client.generate(model='deepseek-R1:7b', prompt=query)
    
    return response['response']

# Test the agent
query = "Say Hello World!"
result = hello_world_agent(query)
print(result)
