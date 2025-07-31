from typing import Any
import runpod

from singleton import init_sentence_transformers_models

def handler(event: dict[str, Any]):
    data = event.get("input")
    return {"data": data}

# Start the Serverless function when the script is run
if __name__ == '__main__':
    init_sentence_transformers_models()
    
    runpod.serverless.start({'handler': handler })