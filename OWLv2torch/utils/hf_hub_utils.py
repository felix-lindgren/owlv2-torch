import os
from pathlib import Path
from huggingface_hub import HfFolder, try_to_load_from_cache
import glob

def find_safetensors_in_cache(model_id: str) -> list[Path]:
    """
    Find all .safetensor files for a specific model in the Hugging Face cache.
    
    Args:
        model_id (str): The Hugging Face model ID (e.g., "facebook/opt-125m")
        
    Returns:
        list[Path]: List of paths to .safetensor files
    """
    # Get the cache directory
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    
    # Convert model_id to cache-friendly format
    # HF stores files in nested directories based on the hash of the model ID
    try:
        cached_file = try_to_load_from_cache(
            model_id,
            "model.safetensors",
            cache_dir=cache_dir
        )
        if cached_file:
            return [Path(cached_file)]
    except Exception:
        pass
    
    # If the above method fails, try manual search
    model_path = Path(cache_dir) / ("models--" + model_id.replace("/", "--"))
    print(model_path)
    if not model_path.exists():
        return []
    
    # Find all .safetensor files recursively
    safetensor_files = list(model_path.rglob("*.safetensors"))
    return safetensor_files
if __name__ == "__main__":
    # Example model ID
    model_id = "google/owlv2-base-patch16-ensemble"
    
    # Find safetensor files
    safetensor_files = find_safetensors_in_cache(model_id)
    print(safetensor_files)
    if not safetensor_files:
        print(f"No .safetensor files found for model {model_id}")
    else:
        print(f"Found {len(safetensor_files)} .safetensor files:")
