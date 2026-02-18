"""
LLM Factory - Free & Popular Model Support
============================================
Supports multiple free tier LLM APIs:
- MiniMax (已配置 via OpenClaw)
- Groq (free llama models)
- Hugging Face Inference (free tier)
- Google Gemini (free tier)
- DeepSeek (competitive free tier)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import json


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        pass
    
    @abstractmethod
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        pass


class MiniMaxProvider(BaseLLMProvider):
    """
    MiniMax API - Already configured in OpenClaw!
    Model: MiniMax-M2.1
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Auto-configure from OpenClaw or environment
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = "https://api.minimax.io/anthropic"
        self.model = "MiniMax-M2.1"
        self.is_available = bool(self.api_key)
        
        if self.is_available:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print("✅ MiniMax (MiniMax-M2.1) configured")
            except ImportError:
                self.is_available = False
                print("⚠️  OpenAI SDK not installed for MiniMax")
        else:
            print("⚠️  MiniMax API key not found (set MINIMAX_API_KEY)")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "MiniMax not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "MiniMax not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class GroqProvider(BaseLLMProvider):
    """
    Groq - Free tier with Llama models!
    https://console.groq.com/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"  # Free tier model
        self.is_available = bool(self.api_key)
        
        if self.is_available:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print("✅ Groq (Llama-3.3-70b) configured")
            except ImportError:
                self.is_available = False
                print("⚠️  OpenAI SDK not installed for Groq")
        else:
            print("⚠️  Groq API key not found (set GROQ_API_KEY)")
            print("   Get free key: https://console.groq.com/")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Groq not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Groq not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Inference - Free tier!
    https://huggingface.co/inference-endpoints
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.model = "meta-llama/Llama-3.3-70B-Instruct"
        self.is_available = bool(self.api_key)
        
        if self.is_available:
            import requests
            self.session = requests.Session()
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            print("✅ Hugging Face (Llama 3.3) configured")
        else:
            print("⚠️  Hugging Face token not found (set HF_TOKEN)")
            print("   Get free token: https://huggingface.co/settings/tokens")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Hugging Face not configured"
        
        import requests
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = self.session.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        return "Hugging Face request failed"
    
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Hugging Face not configured"
        
        # Convert messages to prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        return self.generate(prompt, max_tokens)


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek - Competitive pricing with generous free tier!
    https://platform.deepseek.com/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.is_available = bool(self.api_key)
        
        if self.is_available:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
                print("✅ DeepSeek (DeepSeek-Chat) configured")
            except ImportError:
                self.is_available = False
                print("⚠️  OpenAI SDK not installed for DeepSeek")
        else:
            print("⚠️  DeepSeek API key not found (set DEEPSEEK_API_KEY)")
            print("   Get free key: https://platform.deepseek.com/")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "DeepSeek not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "DeepSeek not configured"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class GoogleGeminiProvider(BaseLLMProvider):
    """
    Google Gemini - Free tier available!
    https://aistudio.google.com/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = "gemini-1.5-flash"  # Free tier
        self.is_available = bool(self.api_key)
        
        if self.is_available:
            print("✅ Google Gemini (1.5 Flash) configured")
        else:
            print("⚠️  Gemini API key not found (set GEMINI_API_KEY)")
            print("   Get free key: https://aistudio.google.com/")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Gemini not configured"
        
        import requests
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = requests.post(url, params=params, json=payload)
        
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return "Gemini request failed"
    
    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.is_available:
            return "Gemini not configured"
        
        # Convert messages to single prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        return self.generate(prompt, max_tokens)


class LLMFactory:
    """
    Factory for creating LLM providers.
    Prioritizes available free models.
    """
    
    # Priority order (first available wins)
    PROVIDERS = [
        ("MiniMax", MiniMaxProvider),
        ("Groq", GroqProvider),
        ("DeepSeek", DeepSeekProvider),
        ("HuggingFace", HuggingFaceProvider),
        ("Gemini", GoogleGeminiProvider),
    ]
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[BaseLLMProvider] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all available providers"""
        for name, provider_class in self.PROVIDERS:
            try:
                provider = provider_class()
                self.providers[name] = provider
                
                if provider.is_available and not self.default_provider:
                    self.default_provider = provider
                    print(f"🎯 Default provider: {name}")
            except Exception as e:
                print(f"⚠️  Failed to initialize {name}: {e}")
    
    def get_default(self) -> Optional[BaseLLMProvider]:
        """Get the default provider"""
        return self.default_provider
    
    def get(self, name: str) -> Optional[BaseLLMProvider]:
        """Get a specific provider"""
        return self.providers.get(name)
    
    def list_available(self) -> list:
        """List available providers"""
        available = []
        for name, provider in self.providers.items():
            if provider.is_available:
                available.append(name)
            else:
                available.append(f"{name} (not configured)")
        return available
    
    def generate(self, prompt: str, provider: Optional[str] = None) -> str:
        """Generate text using specified or default provider"""
        if provider:
            p = self.get(provider)
            if p and p.is_available:
                return p.generate(prompt)
            return f"Provider {provider} not available"
        
        if self.default_provider:
            return self.default_provider.generate(prompt)
        return "No LLM provider available"
    
    def chat(self, messages: list, provider: Optional[str] = None) -> str:
        """Chat using specified or default provider"""
        if provider:
            p = self.get(provider)
            if p and p.is_available:
                return p.chat(messages)
            return f"Provider {provider} not available"
        
        if self.default_provider:
            return self.default_provider.chat(messages)
        return "No LLM provider available"


def demo_llm_factory():
    """Demo LLM factory"""
    print("\n" + "="*60)
    print("🤖 LLM Factory Demo - Free Model Support")
    print("="*60)
    
    factory = LLMFactory()
    
    print(f"\n📋 Available Providers:")
    for name in factory.list_available():
        status = "✅" if "(" not in name else "❌"
        print(f"   {status} {name}")
    
    print(f"\n🎯 Default Provider: {type(factory.get_default()).__name__ if factory.get_default() else 'None'}")
    
    # Test generation
    if factory.get_default():
        print(f"\n🧪 Testing generation:")
        response = factory.generate("What are the top 3 fashion trends for 2025?")
        print(f"   Response: {response[:200]}...")
    
    return factory


# Usage Example
if __name__ == "__main__":
    factory = demo_llm_factory()
    
    print("\n" + "="*60)
    print("📝 Usage Example:")
    print("="*60)
    print("""
from llm_factory import LLMFactory

# Initialize
factory = LLMFactory()

# Generate text
response = factory.generate("Recommend a spring outfit")

# Chat
messages = [
    {"role": "user", "content": "What's your style?"},
    {"role": "assistant", "content": "I love minimalist fashion!"},
    {"role": "user", "content": "Suggest an outfit"}
]
response = factory.chat(messages)
    """)
