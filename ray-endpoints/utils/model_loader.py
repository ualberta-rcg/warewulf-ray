"""
Model loading utilities
"""

import os
from typing import Dict, List, Optional
from pathlib import Path


def list_onnx_models(model_repository: str = "/data/ray-triton/model_repository") -> List[str]:
    """
    List available ONNX models in Triton model repository
    
    Args:
        model_repository: Path to Triton model repository
    
    Returns:
        List of model names
    """
    if not os.path.exists(model_repository):
        return []
    
    models = []
    for item in os.listdir(model_repository):
        model_path = os.path.join(model_repository, item)
        if os.path.isdir(model_path):
            # Check if it has a config.pbtxt (Triton model)
            config_path = os.path.join(model_path, "config.pbtxt")
            if os.path.exists(config_path):
                models.append(item)
    
    return sorted(models)


def list_stable_diffusion_models(models_dir: str = "/data/models/stablediffusion") -> List[Dict[str, str]]:
    """
    List available Stable Diffusion models
    
    Args:
        models_dir: Path to Stable Diffusion models directory
    
    Returns:
        List of model info dictionaries
    """
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        if os.path.isfile(filepath):
            file_ext = Path(filename).suffix
            if file_ext in ['.ckpt', '.safetensors', '.pt']:
                file_size = os.path.getsize(filepath)
                models.append({
                    "name": filename,
                    "path": filepath,
                    "size": file_size,
                    "size_mb": file_size / (1024 * 1024),
                    "type": file_ext[1:]  # Remove the dot
                })
    
    return sorted(models, key=lambda x: x["name"])


def get_model_info(model_name: str, model_repository: str = "/data/ray-triton/model_repository") -> Optional[Dict]:
    """
    Get information about a specific model
    
    Args:
        model_name: Name of the model
        model_repository: Path to Triton model repository
    
    Returns:
        Model information dictionary or None
    """
    model_path = os.path.join(model_repository, model_name)
    if not os.path.exists(model_path):
        return None
    
    info = {
        "name": model_name,
        "path": model_path,
        "config": os.path.join(model_path, "config.pbtxt"),
        "versions": []
    }
    
    # Find model versions
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            info["versions"].append(int(item))
    
    info["versions"].sort(reverse=True)
    info["latest_version"] = info["versions"][0] if info["versions"] else None
    
    return info
