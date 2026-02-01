import requests

#url = "https://unboggy-nonignitible-maira.ngrok-free.dev/webhook-test/local-rag/chat"
url = "http://127.0.0.1:5000/answer"
#url = "http://127.0.0.1:5000/retrieve"

payload = {
    "query": "What are the reported 'cross-media effects' of using 'wet scrubbers' for dust capture in cupola systems compared to dry systems?",
#    "llm_id": "ollama:llama3.2"
    "llm_id": "openai:gpt-4o-mini"
}

r = requests.post(url, json=payload, timeout=120)
r.raise_for_status()

# Debug: see what the API actually returns
response = r.json()
print(f"Response type: {type(response)}")
print(f"Response structure: {response}")
print("-" * 50)

# Handle different response structures
if isinstance(response, list):
    # Response is a list
    if response:
        print(f"First element: {response[0]}")
        # Try to find the answer in the first element
        if isinstance(response[0], dict):
            print(response[0].get("answer") or response[0].get("context") or response[0])
        else:
            print(response[0])
    else:
        print("Empty list response")
elif isinstance(response, dict):
    # Response is a dict (expected)
    print(response.get("answer") or response.get("context") or response)
else:
    print(f"Unexpected response type: {response}")
