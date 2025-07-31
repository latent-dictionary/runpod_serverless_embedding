from typing import Any
import runpod

def handler(event: dict[str, Any]):
    data = event.get("input")
    return {"data": data}

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })