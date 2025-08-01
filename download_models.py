from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Configuration ---
# Specify the models you want to bake into the image
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
# The local directory where models will be stored
LOCAL_MODELS_DIR = "/models"

# --- Main Download Logic ---
if __name__ == "__main__":
    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    snapshot_download(
        repo_id=EMBEDDING_MODEL_NAME,
        local_dir=f"{LOCAL_MODELS_DIR}/{EMBEDDING_MODEL_NAME.replace('/', '_')}",
        local_dir_use_symlinks=False,  # Important for Docker compatibility
        resume_download=True,
    )
    print("Embedding model downloaded.")

    print(f"Downloading reranker model: {RERANKER_MODEL_NAME}")
    snapshot_download(
        repo_id=RERANKER_MODEL_NAME,
        local_dir=f"{LOCAL_MODELS_DIR}/{RERANKER_MODEL_NAME.replace('/', '_')}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Reranker model downloaded.")

    print("All models have been downloaded successfully.")
