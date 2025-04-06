from typing import (List, Optional, Tuple, NamedTuple, Dict, Mapping, MutableMapping, Iterable, Any, Sequence, Set)
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

class Llm(ABC):
    @abstractmethod
    @classmethod
    def _set_models(cls):
        pass

    @abstractmethod
    @classmethod
    def _get_models(cls):
        pass

class LlmUsage(ABC):
    pass

@dataclass
class LLMProvider:  # Changed from plural to singular for better naming
    provider: str
    model_name: str
    default: bool
    latest: bool
    reasoning: bool
    video_generation: bool
    video_analysis: bool
    audio_generation: bool
    audio_analysis: bool

class Gemini(Llm):
    available_models: Dict[str, LLMProvider] = {}
    
    def __init__(self, model_name: str) -> None:
        self.__class__._set_models()
        if model_name is not None and model_name in self.__class__.available_models:
            self._model_name: str = model_name
        else:
            default_model = self.__class__._default_model()
            if default_model is None:
                raise ValueError("No default model available and provided model name is invalid")
            self._model_name: str = default_model

    @classmethod
    def _set_models(cls) -> None:
        # Only populate if not already set
        if cls.available_models:
            return
            
        provider = "Google"
        model_names = [
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001",
            "gemini-2.0-pro-exp-02-05",
            "gemma-3",
            "gemma-2",
            "gemma",
            "codegemma",
            "paligemma-2",
            "paligemma",
            "shieldgemma-2",
            "txgemma"
        ]
    
        # Dictionary to store all model configurations
        models_info = {}
        
        # Process each model and set capabilities based on available information
        for model_name in model_names:
            # Default values
            default = False
            latest = False
            reasoning = False
            video_generation = False
            video_analysis = False
            audio_generation = False
            audio_analysis = False
            
            # Set capabilities based on model name
            match model_name:
                case "gemini-2.5-pro-exp-03-25":
                    latest = True  # Most recent model as of April 2025
                    reasoning = True
                    video_analysis = True
                    audio_analysis = True
                
                case "gemini-2.0-flash":
                    default = True
                    reasoning = True
                    video_analysis = True
                    audio_generation = True
                    audio_analysis = True
                
                case "gemini-2.0-flash-lite":
                    video_analysis = True
                    audio_analysis = True
                
                case "gemini-2.0-flash-thinking-exp-01-21":
                    reasoning = True
                    video_analysis = True
                    audio_analysis = True
                
                case "gemini-2.0-pro-exp-02-05":
                    reasoning = True
                    video_analysis = True
                    audio_analysis = True
                
                case _ if model_name.startswith("gemini-1"):
                    video_analysis = True
                    audio_analysis = True
            
            # Create LLMProvider instance for current model
            models_info[model_name] = LLMProvider(
                provider=provider,
                model_name=model_name,
                default=default,
                latest=latest,
                reasoning=reasoning,
                video_generation=video_generation,
                video_analysis=video_analysis,
                audio_generation=audio_generation,
                audio_analysis=audio_analysis
            )
        
        # Assign to the class variable correctly
        cls.available_models = models_info

    @classmethod
    def _default_model(cls) -> Optional[str]:
        for model_name, model_data in cls.available_models.items():
            if model_data.default:
                return model_name
        return None  # Explicit return None if no default found

    def get_model_name(self) -> str:
        """Return the current model name."""
        return self._model_name
        
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return a list of all available model names."""
        cls._set_models()  # Ensure models are populated
        return list(cls.available_models.keys())
    
#What does a dashboard component have 1. Graph, 2. Stats, 3. Alerts



