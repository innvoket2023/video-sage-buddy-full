from basellm import BaseLLM, ModelInfo, ModelCapabilities, ResponseFormat
from typing import Dict, List, Optional, Tuple, ClassVar, Any
import os
import requests

# Google Gemini Implementation
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
            # This requires handling server-sent events
            raise NotImplementedError("Streaming not implemented for simplicity")
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from Gemini API")
    
    def chat(self, messages: List[Dict[str, str]],
             format: ResponseFormat = ResponseFormat.TEXT,
             stream: bool = False) -> str:
        """Chat with Gemini using messages"""
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
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from Gemini API")
    
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


# OpenAI Implementation
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
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from OpenAI API")
    
    def chat(self, messages: List[Dict[str, str]],
             format: ResponseFormat = ResponseFormat.TEXT,
             stream: bool = False) -> str:
        """Chat with OpenAI using messages"""
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
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from OpenAI API")
    
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
                    max_tokens=4096
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


# Anthropic Implementation
class Anthropic(BaseLLM):
    """Anthropic's Claude LLM implementation"""
    provider_name = "Anthropic"
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variable"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return api_key
    
    def generate(self, prompt: str, 
                 format: ResponseFormat = ResponseFormat.TEXT,
                 stream: bool = False) -> str:
        """Generate text from prompt using Anthropic"""
        api_url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "stream": stream
        }
        
        if self._max_tokens:
            data["max_tokens"] = self._max_tokens
            
        if format == ResponseFormat.JSON:
            data["system"] = "Please provide your response in valid JSON format."
        
        if stream:
            # Implementation for streaming would go here
            raise NotImplementedError("Streaming not implemented for simplicity")
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        try:
            return result["content"][0]["text"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from Anthropic API")
    
    def chat(self, messages: List[Dict[str, str]],
             format: ResponseFormat = ResponseFormat.TEXT,
             stream: bool = False) -> str:
        """Chat with Anthropic using messages"""
        api_url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Format system message separately for Anthropic
        system_message = None
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                system_message = content
            else:
                anthropic_role = "user" if role == "user" else "assistant"
                formatted_messages.append({"role": anthropic_role, "content": content})
        
        data = {
            "model": self._model_name,
            "messages": formatted_messages,
            "temperature": self._temperature,
            "stream": stream
        }
        
        if system_message:
            data["system"] = system_message
            
        if self._max_tokens:
            data["max_tokens"] = self._max_tokens
            
        if format == ResponseFormat.JSON and not system_message:
            data["system"] = "Please provide your response in valid JSON format."
        elif format == ResponseFormat.JSON and system_message:
            data["system"] += " Please provide your response in valid JSON format."
        
        if stream:
            # Implementation for streaming would go here
            raise NotImplementedError("Streaming not implemented for simplicity")
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        try:
            return result["content"][0]["text"]
        except (KeyError, IndexError):
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
            raise ValueError("Unexpected response format from Anthropic API")
    
    @classmethod
    def _get_model_data(cls) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Provide Anthropic-specific model data"""
        provider = "Anthropic"
        
        model_data = {
            "claude-3-5-sonnet-20240620": {
                "capabilities": ModelCapabilities(
                    default=True,
                    latest=True,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=200000
                )
            },
            "claude-3-sonnet-20240229": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=180000
                )
            },
            "claude-3-opus-20240229": {
                "capabilities": ModelCapabilities(
                    default=False,
                    latest=False,
                    reasoning=True,
                    video_generation=False,
                    video_analysis=True,
                    audio_generation=False,
                    audio_analysis=True,
                    max_tokens=180000
                )
            }
        }
        
        return provider, model_data

