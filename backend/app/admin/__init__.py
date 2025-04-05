from typing import (List, Tuple, NamedTuple, Dict, Mapping, MutableMapping, Iterable, Any, Sequence, Set)
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

class Llm(ABC):
    pass

class LlmUsage(ABC):
    pass

@dataclass
class LLMProviders():
    name: str
    capabilities: Set[str]

class Gemini(Llm):
    available_models: Set[] = {}
    def __init__(self, model_name: str) -> None:
        if model_name is not None and model_name in available_models:
            self._model_name:str = model_name
#What does a dashboard component have 1. Graph, 2. Stats, 3. Alerts



