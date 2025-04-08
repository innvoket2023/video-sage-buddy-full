from app.admin.llmusage import LLMUsage
from typing import Dict, List, Optional, Tuple, ClassVar, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import json
import requests
from enum import Enum

# Core data structures
@dataclass
class ModelCapabilities:
    default: bool = False
    latest: bool = False
    reasoning: bool = False
    video_generation: bool = False
    video_analysis: bool = False
    audio_generation: bool = False
    audio_analysis: bool = False
    max_tokens: int = 4096

@dataclass
class ModelInfo:
    provider: str
    model_name: str
    capabilities: ModelCapabilities

class ResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"

# Base LLM class that all providers will inherit from
class BaseLLM(ABC):
    """Base class implementing common LLM functionality"""
    provider_name: ClassVar[str]  # Class variable to be set by subclasses
    available_models: Dict[str, ModelInfo] = {}
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 usage_tracker: Optional[LLMUsage] = None) -> None:
        """Initialize LLM with specified model or default"""
        self._api_key = api_key or self._get_api_key_from_env()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.usage_tracker = usage_tracker
        
        # Initialize models data if not already done
        if not self.__class__.available_models:
            self.__class__._set_models()
        
        # Set the model name
        if model_name and model_name in self.__class__.available_models:
            self._model_name = model_name
        else:
            self._model_name = self._get_default_model_name()
            if not self._model_name:
                raise ValueError(f"No default model available for {self.provider_name}")
    
    @classmethod
    def _set_models(cls) -> None:
        """Set up available models if not already populated"""
        if cls.available_models:
            return
            
        # Get model data from child class
        provider, model_data = cls._get_model_data()
        
        # Dictionary to store model configurations
        models_info = {}
        
        # Process each model
        for model_name, capabilities in model_data.items():
            models_info[model_name] = ModelInfo(
                provider=provider,
                model_name=model_name,
                **capabilities
            )
        
        # Assign to the class variable
        cls.available_models = models_info
    
    @abstractmethod
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variable"""
        pass
        
    @classmethod
    def _get_default_model_name(cls) -> Optional[str]:
        """Find default model"""
        for model_name, model_info in cls.available_models.items():
            if model_info.capabilities.default:
                return model_name
        return None
    
    @abstractmethod
    def generate(self, prompt: str,
                 format: ResponseFormat = ResponseFormat.TEXT,
                 stream: bool = False,
                 user_id: Optional[str] = None) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             format: ResponseFormat = ResponseFormat.TEXT,
             stream: bool = False,
             user_id: Optional[str] = None) -> str:
        """Chat with the model using messages"""
        pass
        
    def get_model_name(self) -> str:
        """Return current model name"""
        return self._model_name
        
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available models"""
        cls._set_models()
        return list(cls.available_models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: Optional[str] = None) -> ModelInfo:
        """Get info for a specific model"""
        cls._set_models()
        if model_name is None:
            raise ValueError("Model name must be provided")
        if model_name not in cls.available_models:
            raise ValueError(f"Model {model_name} not found")
        return cls.available_models[model_name]

    @classmethod
    @abstractmethod
    def _get_model_data(cls) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Child classes must implement this to provide model data"""
        pass
