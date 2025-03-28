from typing import List, Optional, Any, Dict, Union
import os
import json
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, 'config', '.env')
load_dotenv(env_path)

class OpenAICompatibleModel:
    """Unified model client for OpenAI-compatible APIs (OpenAI, Gemini, Mistral)."""

    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": os.getenv("OPENAI_BASE_URL", "https://chat-api.inovex.ai/"),
            "api_key": os.getenv("OPENAI_API_KEY", None)
        },
        "gemini": {
            "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            "api_key": os.getenv("GEMINI_API_KEY", None)
        },
        "mistral": {
            "base_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/"),
            "api_key": os.getenv("MISTRAL_API_KEY", None)
        }
    }

    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "mistral": "pixtral-large"
    }

    def __init__(self, provider: str, model_name: Optional[str] = None):
        """
        Initialize the model client.

        Args:
            provider: Provider name ("openai", "gemini", or "mistral")
            model_name: Optional model name (if None, uses the default for the provider)
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: {', '.join(self.PROVIDER_CONFIGS.keys())}")

        self.provider = provider
        self.config = self.PROVIDER_CONFIGS[provider]
        self.model_name = model_name or self.DEFAULT_MODELS[provider]
        self.client = self.initialize_client()



    def initialize_client(self) -> Any:
        """
        Initialize the OpenAI client for the chosen provider.

        Returns:
            OpenAI client instance
        """
        return OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
