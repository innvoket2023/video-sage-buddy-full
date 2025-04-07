import os
import json
import uuid
import time
import requests
from typing import Dict, List, Optional, Tuple, Any

from basellm import BaseLLM, ResponseFormat, ModelCapabilities

class OpenAI(BaseLLM):
    """OpenAI LLM implementation"""
    provider_name = "OpenAI"
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variable"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key
    
    def generate(self, prompt: str, 
                 format: ResponseFormat = ResponseFormat.TEXT,
                 stream: bool = False) -> str:
        """Generate text from prompt using OpenAI"""
        # Track request with unique identifier
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            api_url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            
            messages = [{"role": "user", "content": prompt}]
            
            data = {
                "model": self._model_name,
                "messages": messages,
                "temperature": self._temperature,
                "stream": stream
            }
            
            if self._max_tokens:
                data["max_tokens"] = self._max_tokens
                
            if format == ResponseFormat.JSON:
                data["response_format"] = {"type": "json_object"}
            
            if stream:
                # Implementation for streaming would go here
                raise NotImplementedError("Streaming not implemented for simplicity")
            
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text
            response_text = self._extract_response_text(result)
            
            # Record usage if tracker is available
            if self.usage_tracker:
                prompt_tokens = self._extract_prompt_tokens(result)
                completion_tokens = self._extract_completion_tokens(result)
                cost = self._calculate_cost(prompt_tokens, completion_tokens)
                
                self.usage_tracker.record_request(
                    request_id=request_id,
                    model=self._model_name,
                    provider=self.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=(time.time() - start_time) * 1000,
                    status="success",
                    cost=cost
                )
            
            return response_text
            
        except Exception as e:
            # Record error if tracker is available
            if self.usage_tracker:
                self.usage_tracker.record_request(
                    request_id=request_id,
                    model=self._model_name,
                    provider=self.provider_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    status="error",
                    error=str(e)
                )
            raise
    
    def chat(self, messages: List[Dict[str, str]],
             format: ResponseFormat = ResponseFormat.TEXT,
             stream: bool = False) -> str:
        """Chat with OpenAI using messages"""
        # Track request with unique identifier
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            api_url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            
            data = {
                "model": self._model_name,
                "messages": messages,
                "temperature": self._temperature,
                "stream": stream
            }
            
            if self._max_tokens:
                data["max_tokens"] = self._max_tokens
                
            if format == ResponseFormat.JSON:
                data["response_format"] = {"type": "json_object"}
            
            if stream:
                # Implementation for streaming would go here
                raise NotImplementedError("Streaming not implemented for simplicity")
            
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text
            response_text = self._extract_response_text(result)
            
            # Record usage if tracker is available
            if self.usage_tracker:
                prompt_tokens = self._extract_prompt_tokens(result)
                completion_tokens = self._extract_completion_tokens(result)
                cost = self._calculate_cost(prompt_tokens, completion_tokens)
                
                self.usage_tracker.record_request(
                    request_id=request_id,
                    model=self._model_name,
                    provider=self.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=(time.time() - start_time) * 1000,
                    status="success",
                    cost=cost
                )
            
            return response_text
            
        except Exception as e:
            # Record error if tracker is available
            if self.usage_tracker:
                self.usage_tracker.record_request(
                    request_id=request_id,
                    model=self._model_name,
                    provider=self.provider_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    status="error",
                    error=str(e)
                )
            raise
    
    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Extract text from OpenAI API response"""
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from OpenAI API")
    
    def _extract_prompt_tokens(self, result: Dict[str, Any]) -> int:
        """Extract prompt tokens from API response"""
        return result.get("usage", {}).get("prompt_tokens", 0)
    
    def _extract_completion_tokens(self, result: Dict[str, Any]) -> int:
        """Extract completion tokens from API response"""
        return result.get("usage", {}).get("completion_tokens", 0)
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model and token usage"""
        # Model pricing (as of April 2025)
        prices = {
            "gpt-4o": {"prompt": 0.00001, "completion": 0.00003},
            "gpt-4-turbo": {"prompt": 0.00001, "completion": 0.00003},
            "gpt-3.5-turbo": {"prompt": 0.0000015, "completion": 0.000002}
        }
        
        model_prices = prices.get(
            self._model_name, 
            {"prompt": 0.00001, "completion": 0.00003}  # Default fallback pricing
        )
        
        return (prompt_tokens * model_prices["prompt"]) + (completion_tokens * model_prices["completion"])
    
    @classmethod
    def _get_model_data(cls) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Provide OpenAI-specific model data"""
        provider = "OpenAI"
        
        model_data = {
            "gpt-4o": {
                "capabilities": ModelCapabilities(
                    default=True,
                    latest=True,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=8192
                )
            },
            "gpt-4-turbo": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=False,
                    audio_generation=False,
                    audio_analysis=False,
                    max_tokens=4096
                )
            },
            "gpt-3.5-turbo": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=False,
                    video_generation=False,
                    video_analysis=False,
                    audio_generation=False,
                    audio_analysis=False,
                    max_tokens=4096
                )
            }
        }
        
        return provider, model_data
