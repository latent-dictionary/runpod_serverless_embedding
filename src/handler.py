import runpod
import time
import threading
from sentence_transformers import SentenceTransformer
from models import embedding_models

from singleton import init_sentence_transformers_models


# --- Global State for Batching ---
# This state will persist across handler calls for a single worker
request_queue = []
# A lock is crucial to prevent race conditions when multiple threads access the queue
queue_lock = threading.Lock()
# A dictionary to store results for each request
results: dict[str, embedding_models.OpenAITextEmbeddingResponse] = {}
# An event to signal when a request is done
request_events: dict[str, threading.Event] = {}

# Load the model ONCE during initialization
model = SentenceTransformer("all-MiniLM-L6-v2")
MAX_BATCH_SIZE = 32
MAX_WAIT_TIME_MS = 20  # 20 milliseconds


# --- The Batch Processing Worker Thread ---
def inference_loop():
    while True:
        requests_to_process = []

        with queue_lock:
            # Wait for requests to come in
            if not request_queue:
                time.sleep(0.001)  # Sleep for 1ms to prevent a busy loop
                continue

            # Start a timer
            start_time = time.time()

            # Wait until batch is full or timer runs out
            while (
                time.time() - start_time < MAX_WAIT_TIME_MS / 1000.0
                and len(request_queue) < MAX_BATCH_SIZE
            ):
                time.sleep(0.001)

            # Grab all requests currently in the queue
            num_to_process = min(len(request_queue), MAX_BATCH_SIZE)
            requests_to_process = request_queue[:num_to_process]
            del request_queue[:num_to_process]

        if requests_to_process:
            # Batch the inputs
            all_texts = [req["input"]["text"] for req in requests_to_process]
            request_ids = [req["id"] for req in requests_to_process]

            # Run inference on the whole batch
            print(f"Processing batch of size {len(all_texts)}")
            embeddings = model.encode(all_texts, convert_to_numpy=True).tolist()

            # De-batch and signal completion
            for i, request_id in enumerate(request_ids):
                results[request_id] = embeddings[i]
                request_events[request_id].set()  # Signal that this request is done


# --- The RunPod Handler ---
def handler(job):
    """
    This is the function that RunPod calls for each request.
    It just adds the job to a queue and waits.
    """
    job_id = job["id"]

    # Add the job to our queue
    with queue_lock:
        # Create an event that this handler can wait on
        request_events[job_id] = threading.Event()
        request_queue.append(job)

    # Wait until the background thread processes this job
    request_events[job_id].wait()

    # Retrieve the result and clean up
    result = results.pop(job_id)
    del request_events[job_id]

    return result


# Start the Serverless function when the script is run
if __name__ == "__main__":
    init_sentence_transformers_models()

    # Start the background inference thread when the worker initializes
    threading.Thread(target=inference_loop, daemon=True).start()

    runpod.serverless.start({"handler": handler})
