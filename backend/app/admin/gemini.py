import os
import json
import uuid
import time
import requests
from typing import Dict, List, Optional, Tuple, Any

from basellm import BaseLLM, ResponseFormat, ModelCapabilities

class Gemini(BaseLLM):
    """Google's Gemini LLM implementation"""
    provider_name = "Google"
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variable"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return api_key
    
    def generate(self, prompt: str, 
                 format: ResponseFormat = ResponseFormat.TEXT,
                 stream: bool = False) -> str:
        """Generate text from prompt using Gemini"""
        # Track request with unique identifier
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key
            }
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self._temperature,
                    "maxOutputTokens": self._max_tokens or 
                        self.available_models[self._model_name].capabilities.max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            if format == ResponseFormat.JSON:
                data["generationConfig"]["responseFormat"] = {"type": "JSON"}
            
            if stream:
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:streamGenerateContent"
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
        """Chat with Gemini using messages"""
        # Track request with unique identifier
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key
            }
            
            # Format messages for Gemini API
            contents = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    # Gemini doesn't have system messages, prepend to first user message
                    if contents:
                        if "parts" in contents[0]:
                            contents[0]["parts"].insert(0, {"text": f"System: {content}\n\n"})
                    else:
                        contents.append({"role": "user", "parts": [{"text": f"System: {content}\n\n"}]})
                else:
                    gemini_role = "user" if role == "user" else "model"
                    contents.append({"role": gemini_role, "parts": [{"text": content}]})
            
            data = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self._temperature,
                    "maxOutputTokens": self._max_tokens or 
                        self.available_models[self._model_name].capabilities.max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            if format == ResponseFormat.JSON:
                data["generationConfig"]["responseFormat"] = {"type": "JSON"}
            
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
        """Extract text from Gemini API response"""
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from Gemini API")
    
    def _extract_prompt_tokens(self, result: Dict[str, Any]) -> int:
        """Extract prompt tokens from API response"""
        try:
            return result.get("usageMetadata", {}).get("promptTokenCount", 0)
        except (KeyError, AttributeError):
            # Estimate based on text length if not provided by API
            # Rough estimate: 1 token per 4 characters
            prompt_text = result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
            return len(prompt_text) // 4 
    
    def _extract_completion_tokens(self, result: Dict[str, Any]) -> int:
        """Extract completion tokens from API response"""
        try:
            return result.get("usageMetadata", {}).get("candidatesTokenCount", 0)
        except (KeyError, AttributeError):
            # Estimate based on response length if not provided by API
            response_text = self._extract_response_text(result)
            return len(response_text) // 4
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model and token usage"""
        # Model pricing (as of April 2025)
        prices = {
            "gemini-2.5-pro-exp-03-25": {"prompt": 0.000014, "completion": 0.000042},
            "gemini-2.0-flash": {"prompt": 0.000007, "completion": 0.000021},
            "gemini-2.0-flash-lite": {"prompt": 0.000004, "completion": 0.000012},
            "gemini-1.5-pro-001": {"prompt": 0.000014, "completion": 0.000042},
            "gemini-1.5-flash-001": {"prompt": 0.000007, "completion": 0.000021},
            "gemini-1.0-pro": {"prompt": 0.000010, "completion": 0.000030}
        }
        
        model_prices = prices.get(
            self._model_name, 
            {"prompt": 0.000010, "completion": 0.000030}  # Default fallback pricing
        )
        
        return (prompt_tokens * model_prices["prompt"]) + (completion_tokens * model_prices["completion"])
    
    @classmethod
    def _get_model_data(cls) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Provide Gemini-specific model data"""
        provider = "Google"
        
        model_data = {
            "gemini-2.5-pro-exp-03-25": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=True,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=8192
                )
            },
            "gemini-2.0-flash": {
                "capabilities": ModelCapabilities(
                    default=True,
                    latest=False,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=True,
                    audio_analysis=True,
                    max_tokens=8192
                )
            },
            "gemini-2.0-flash-lite": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=False,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=4096
                )
            },
            "gemini-1.5-pro-001": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=4096
                )
            },
            "gemini-1.5-flash-001": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=False,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=4096
                )
            },
            "gemini-1.0-pro": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=False,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=4096
                )
            }
        }
        
        return provider, model_data
