from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

def get_model(**kwargs):
    """Load HuggingFace pretrained model"""
    if kwargs.pop("model_parallel")==True:
        model = AutoModelForCausalLM.from_pretrained(device_map="auto", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    return model


def get_custom_model(model_type, **kwargs):
    """
    Load custom model classes.
    
    Args:
        model_type (str): Type of custom model to load
            - "tiny_recursive": TinyRecursiveModel
        **kwargs: Model-specific parameters
        
    Returns:
        Instantiated model
    """
    if model_type == "tiny_recursive":
        from .tiny_recursive_model import TinyRecursiveModel
        supported_kwargs = {
            "vocab_size",
            "dim",
            "n_heads",
            "n_layers",
            "mlp_ratio",
            "max_seq_len",
            "n_latent_recursions",
            "n_improvement_cycles",
        }
        clean_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        # Ensure vocab_size is provided
        if "vocab_size" not in clean_kwargs:
            raise ValueError("vocab_size is required for TinyRecursiveModel. Pass it via config or command line.")
        return TinyRecursiveModel(**clean_kwargs)
    else:
        raise ValueError(f"Unknown custom model type: {model_type}")


def get_model_by_name(model_name, model_type="huggingface", **kwargs):
    """
    Unified interface to load models by name.
    
    Args:
        model_name (str): Model identifier or custom model type
        model_type (str): "huggingface" or "custom"
        **kwargs: Additional model parameters
        
    Returns:
        Instantiated model
    """
    if model_type == "huggingface":
        return get_model(pretrained_model_name_or_path=model_name, **kwargs)
    elif model_type == "custom":
        return get_custom_model(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
