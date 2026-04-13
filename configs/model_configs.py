"""
Model configurations for different LLMs in GuardRAG evaluation.

Author: Gayatri Malladi
"""

MODEL_CONFIGS = {
    # Llama 3.1 models (local cluster paths — offline mode)
    'llama-3.1-8b': {
        'model_name': '/gscratch/uwb/gayat23/models/Meta-Llama-3-8B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': True,   # 8-bit for memory savings on RTX 6000
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Llama 3.1 8B Instruct (local)',
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
        'cpu_offload': True,
        'max_gpu_memory': '44GiB',
        'max_cpu_memory': '120GiB',
        'offload_folder': '/gscratch/uwb/gayat23/GuardRAG/cache/offload/llama-3.1-70b-4bit',
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
        'model_name': '/gscratch/uwb/gayat23/hf_cache/Llama-3.3-70B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': True,  # 4-bit quantization
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',
        'cpu_offload': True,
        'max_gpu_memory': '44GiB',
        'max_cpu_memory': '120GiB',
        'offload_folder': '/gscratch/uwb/gayat23/GuardRAG/cache/offload/llama-3.3-70b-4bit',
        'description': 'Llama 3.3 70B Instruct (4-bit, ~35GB) - local',
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

    # ── Mistral models (local, open-source) ─────────────────────────────────
    'mistral-7b': {
        'model_name': '/gscratch/uwb/gayat23/hf_cache/Mistral-7B-Instruct-v0.3',
        'provider': 'local',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': True,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Mistral 7B Instruct v0.3 (local, 8-bit, ~7GB VRAM)',
        'min_gpu_memory': '16GB',
    },

    # ── Qwen models (local) ──────────────────────────────────────────────────
    'qwen3.5-27b': {
        'model_name': '/gscratch/uwb/gayat23/hf_cache/Qwen3.5-27B',
        'provider': 'local',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': True,
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',
        'description': 'Qwen3.5 27B (local, 4-bit, ~14GB VRAM)',
        'min_gpu_memory': '16GB',
        'trust_remote_code': True,
    },

    'glm-4-9b-chat-hf': {
        'model_name': '/gscratch/uwb/gayat23/hf_cache/glm-4-9b-chat-hf',
        'provider': 'local',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': False,
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',
        'trust_remote_code': True,
        'use_chat_template': True,
        'description': 'GLM-4-9B-Chat HF (local, BF16, chat-based GLM baseline)',
        'min_gpu_memory': '24GB',
    },

    'mixtral-8x7b': {
        'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'provider': 'local',
        'max_new_tokens': 256,
        'temperature': 0.0,
        'top_p': 1.0,
        'device': 'cuda',
        'load_in_8bit': False,
        'load_in_4bit': True,
        'device_map': 'auto',
        'torch_dtype': 'float16',
        'description': 'Mixtral 8x7B Instruct (4-bit quantized, ~24GB VRAM)',
        'min_gpu_memory': '24GB',
    },

    # ── OpenAI API models ────────────────────────────────────────────────────
    'gpt-5.2': {
        'model_name': 'gpt-5.2',
        'provider': 'openai',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'GPT-5.2 (OpenAI API) — flagship commercial baseline',
    },

    'gpt-5-mini': {
        'model_name': 'gpt-5-mini',
        'provider': 'openai',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'GPT-5 Mini (OpenAI API) — efficient commercial baseline',
    },

    'gpt-4o': {
        'model_name': 'gpt-4o',
        'provider': 'openai',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'GPT-4o (OpenAI API) — previous generation commercial baseline',
    },

    'gpt-4o-mini': {
        'model_name': 'gpt-4o-mini',
        'provider': 'openai',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'GPT-4o Mini (OpenAI API) — efficient baseline',
    },

    # ── Anthropic API models ─────────────────────────────────────────────────
    'claude-3-5-sonnet': {
        'model_name': 'claude-sonnet-4-6',
        'provider': 'anthropic',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'Claude Sonnet 4.6 (Anthropic API)',
    },

    'claude-3-5-haiku': {
        'model_name': 'claude-haiku-4-5-20251001',
        'provider': 'anthropic',
        'max_new_tokens': 512,
        'temperature': 0.0,
        'top_p': 1.0,
        'description': 'Claude Haiku 4.5 (Anthropic API) — efficient',
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
