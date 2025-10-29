"""
Configuration file for LLM settings
Easy to switch between different providers
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Provider Configuration
# Options: 'gemini', 'openai', 'anthropic', 'groq', 'openrouter', 'qwen', 'huggingface'
# Default to Gemini - most reliable free option
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')

# API Keys (set via environment variables)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
QWEN_API_URL = os.getenv('QWEN_API_URL', 'https://wwatashi84--qwen3-8b-vllm-serve.modal.run/v1')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')

# Model configurations
LLM_MODELS = {
    'gemini': {
        'model': 'gemini-2.0-flash-exp',  # Fast and free
        'temperature': 0.7,
    },
    'openai': {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
    },
    'anthropic': {
        'model': 'claude-3-haiku-20240307',
        'temperature': 0.7,
    },
    'groq': {
        'model': 'llama-3.1-70b-versatile',
        'temperature': 0.7,
    },
    'openrouter': {
        'model': 'qwen/qwen-2.5-7b-instruct:free',  # Newer Qwen 2.5 - Free and better!
        'temperature': 0.7,
    },
    'qwen': {
        'model': 'qwen2.5:3b',  # Qwen 2.5 3B running locally with Ollama - 100% FREE!
        'temperature': 0.7,
    },
    'huggingface': {
        'model': 'mistralai/Mistral-7B-Instruct-v0.2',  # FREE GPU-powered Mistral on Hugging Face!
        'temperature': 0.7,
    }
}

# Get current model config
CURRENT_MODEL_CONFIG = LLM_MODELS.get(LLM_PROVIDER, LLM_MODELS['gemini'])
