"""
LLM providers module for the Airline Flight Insights Assistant.
Supports multiple LLM providers: Google Gemini, Groq, and HuggingFace.
"""

import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import GOOGLE_API_KEY, GROQ_API_KEY, HUGGINGFACE_API_TOKEN


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    provider: str
    response_time: float
    tokens_estimate: int
    success: bool
    error: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini model."""
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0
                )
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=self.model,
                provider="google",
                response_time=0,
                tokens_estimate=0,
                success=False,
                error="Gemini not available. Check GOOGLE_API_KEY."
            )
        
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time() - start_time
            content = response.content if hasattr(response, 'content') else str(response)
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="google",
                response_time=elapsed,
                tokens_estimate=len(content.split()),
                success=True
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="google",
                response_time=time.time() - start_time,
                tokens_estimate=0,
                success=False,
                error=str(e)
            )


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider (Llama, Mixtral, etc.)."""
    
    def __init__(self, model: str = "llama3-8b-8192"):
        self.model = model
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Groq model."""
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model=self.model,
                    api_key=GROQ_API_KEY,
                    temperature=0
                )
            except Exception as e:
                print(f"Failed to initialize Groq: {e}")
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=self.model,
                provider="groq",
                response_time=0,
                tokens_estimate=0,
                success=False,
                error="Groq not available. Check GROQ API key."
            )
        
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time() - start_time
            content = response.content if hasattr(response, 'content') else str(response)
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="groq",
                response_time=elapsed,
                tokens_estimate=len(content.split()),
                success=True
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="groq",
                response_time=time.time() - start_time,
                tokens_estimate=0,
                success=False,
                error=str(e)
            )


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider."""
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model = model
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the HuggingFace model."""
        if HUGGINGFACE_API_TOKEN:
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                endpoint = HuggingFaceEndpoint(
                    repo_id=self.model,
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                )
                self.llm = ChatHuggingFace(llm=endpoint)
            except Exception as e:
                print(f"Failed to initialize HuggingFace: {e}")
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=self.model,
                provider="huggingface",
                response_time=0,
                tokens_estimate=0,
                success=False,
                error="HuggingFace not available. Check API token."
            )
        
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time() - start_time
            content = response.content if hasattr(response, 'content') else str(response)
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="huggingface",
                response_time=elapsed,
                tokens_estimate=len(content.split()),
                success=True
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="huggingface",
                response_time=time.time() - start_time,
                tokens_estimate=0,
                success=False,
                error=str(e)
            )


# Model name to provider mapping
MODEL_PROVIDERS = {
    "Gemini-2.0-Flash": ("google", "gemini-2.0-flash"),
    "Gemini-1.5-Flash": ("google", "gemini-1.5-flash"),
    "Llama-3-8B": ("groq", "llama3-8b-8192"),
    "Llama-3-70B": ("groq", "llama3-70b-8192"),
    "Mixtral-8x7B": ("groq", "mixtral-8x7b-32768"),
    "Mistral-7B": ("huggingface", "mistralai/Mistral-7B-Instruct-v0.2"),
}


class LLMManager:
    """Manager class for handling multiple LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
    
    def get_provider(self, model_name: str) -> Optional[BaseLLMProvider]:
        """Get or create a provider for the specified model."""
        if model_name in self._providers:
            return self._providers[model_name]
        
        if model_name not in MODEL_PROVIDERS:
            return None
        
        provider_type, model_id = MODEL_PROVIDERS[model_name]
        
        if provider_type == "google":
            provider = GeminiProvider(model_id)
        elif provider_type == "groq":
            provider = GroqProvider(model_id)
        elif provider_type == "huggingface":
            provider = HuggingFaceProvider(model_id)
        else:
            return None
        
        if provider.is_available():
            self._providers[model_name] = provider
            return provider
        
        return None
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using the specified model."""
        provider = self.get_provider(model_name)
        
        if provider is None:
            return LLMResponse(
                content="",
                model=model_name,
                provider="unknown",
                response_time=0,
                tokens_estimate=0,
                success=False,
                error=f"Model '{model_name}' not available."
            )
        
        return provider.generate(prompt, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        available = []
        for model_name in MODEL_PROVIDERS.keys():
            provider = self.get_provider(model_name)
            if provider and provider.is_available():
                available.append(model_name)
        return available


# Global LLM manager instance
llm_manager = LLMManager()


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    return llm_manager


def generate_response(model_name: str, prompt: str, **kwargs) -> LLMResponse:
    """Convenience function to generate a response."""
    return llm_manager.generate(model_name, prompt, **kwargs)
