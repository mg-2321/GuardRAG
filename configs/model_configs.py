"""
Model configurations for different LLMs in GuardRAG evaluation.
"""

MODEL_CONFIGS = {
    # Llama 3.1 models (older)
    'llama-3.1-8b': {
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Llama 3.1 8B Instruct (baseline model from Phase 1)',
    },
    
    'llama-3.1-70b': {
        'model_name': 'meta-llama/Llama-3.1-70B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Llama 3.1 70B Instruct (older)',
        'min_gpu_memory': '160GB',
        'recommended_gpu': '4x A100 80GB',
    },
    
    'llama-3.1-70b-4bit': {
        'model_name': 'meta-llama/Llama-3.1-70B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': True,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Llama 3.1 70B Instruct (4-bit quantized, ~40GB memory)',
        'min_gpu_memory': '48GB',
    },
    
    # Llama 3.3 models (NEWEST - December 2024)
    'llama-3.3-70b': {
        'model_name': 'meta-llama/Llama-3.3-70B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',  # 3.3 prefers bfloat16
        'description': 'Llama 3.3 70B Instruct (NEWEST, Dec 2024 - Best performance)',
        'min_gpu_memory': '160GB',
        'recommended_gpu': '4x A100 80GB',
    },
    
    'llama-3.3-70b-4bit': {
        'model_name': 'meta-llama/Llama-3.3-70B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': True,  # 4-bit quantization
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',
        'description': 'Llama 3.3 70B Instruct (4-bit, ~40GB) - RECOMMENDED for RTX 6000',
        'min_gpu_memory': '48GB',
    },
    
    # Llama 2 models (for comparison)
    'llama-2-70b': {
        'model_name': 'meta-llama/Llama-2-70b-chat-hf',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Llama 2 70B Chat (comparison baseline)',
    },
}

def get_model_config(model_key: str):
    """Get model configuration by key"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_key}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]

def list_models():
    """List all available model configurations"""
    print("Available Models:")
    print("=" * 80)
    for key, config in MODEL_CONFIGS.items():
        print(f"\n{key}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Description: {config['description']}")
        if 'min_gpu_memory' in config:
            print(f"  Min GPU Memory: {config['min_gpu_memory']}")
        if 'recommended_gpu' in config:
            print(f"  Recommended: {config['recommended_gpu']}")

if __name__ == '__main__':
    list_models()
