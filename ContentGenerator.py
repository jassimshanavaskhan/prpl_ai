import time
import json
import logging
from typing import Optional, Dict, Any, List, Union
from enum import Enum, auto
from dataclasses import dataclass
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import groq
# from anthropic import Anthropic
import os
import re
import requests  # Added for LM Studio API

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("AdvancedContentGenerator")
from logger import logger

class ModelProvider(Enum):
    GEMINI = auto()
    GROQ = auto()
    LMSTUDIO = auto()  # Added LM Studio provider
    # ANTHROPIC = auto()
    # OPENAI = auto()  # For future expansion

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    base_url: Optional[str] = None  # Added for LM Studio API base URL
    
    def __post_init__(self):
        # Validate temperature
        if not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")

@dataclass
class GenerationResponse:
    text: str
    model_used: str
    provider_used: ModelProvider
    is_fallback: bool
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None
    
    def __str__(self):
        return self.text

class AdvancedContentGenerator:
    """
    Advanced content generation with multiple model support, retries, and fallbacks
    """
    
    def __init__(self, 
                 primary_config: ModelConfig,
                 fallback_configs: Optional[List[ModelConfig]] = None,
                 default_max_retries: int = 5,
                 initial_retry_delay: float = 1.0,
                 request_timeout: int = 60):
        """
        Initialize the content generator with primary and fallback models.
        
        Args:
            primary_config: Configuration for the primary model
            fallback_configs: List of configurations for fallback models (in priority order)
            default_max_retries: Default maximum number of retries for each model
            initial_retry_delay: Initial delay between retries (will increase exponentially)
            request_timeout: Timeout for API requests in seconds
        """
        self.primary_config = primary_config
        self.fallback_configs = fallback_configs or []
        self.default_max_retries = default_max_retries
        self.initial_retry_delay = initial_retry_delay
        self.request_timeout = request_timeout
        
        # Initialize clients for each unique provider
        self.clients = {}
        self._initialize_clients([primary_config] + self.fallback_configs)
        
        # Track usage and rate limits
        self.usage_stats = {provider: {"requests": 0, "last_request_time": 0} 
                           for provider in ModelProvider}
    
    def _initialize_clients(self, configs: List[ModelConfig]):
        """Initialize API clients for each unique provider"""
        for config in configs:
            if config.provider not in self.clients:
                if config.provider == ModelProvider.GEMINI:
                    genai.configure(api_key=config.api_key)
                    self.clients[config.provider] = {"configured": True}
                
                elif config.provider == ModelProvider.GROQ:
                    self.clients[config.provider] = {
                        "client": groq.Client(api_key=config.api_key)
                    }
                
                elif config.provider == ModelProvider.LMSTUDIO:
                    # LM Studio API often uses a local server setup
                    # We'll store the base URL (default to localhost if not provided)
                    base_url = config.base_url or "http://localhost:1234/v1"
                    self.clients[config.provider] = {
                        "base_url": base_url,
                        "api_key": config.api_key  # might be empty for local LM Studio
                    }
                
                # elif config.provider == ModelProvider.ANTHROPIC:
                #     self.clients[config.provider] = {
                #         "client": Anthropic(api_key=config.api_key)
                #     }
    
    def _update_usage_stats(self, provider: ModelProvider):
        """Update usage statistics for rate limiting"""
        current_time = time.time()
        self.usage_stats[provider]["requests"] += 1
        self.usage_stats[provider]["last_request_time"] = current_time
    
    def _should_rate_limit(self, provider: ModelProvider, requests_per_minute: int = 60) -> bool:
        """Check if we should rate limit based on recent usage"""
        current_time = time.time()
        stats = self.usage_stats[provider]
        
        # If less than a minute has passed since first request and we've hit the limit
        if (current_time - stats["last_request_time"] < 60 and 
            stats["requests"] >= requests_per_minute):
            return True
        
        # Reset counter if more than a minute has passed
        if current_time - stats["last_request_time"] >= 60:
            stats["requests"] = 0
            
        return False
    
    def generate_content(self, 
                         prompt: str, 
                         max_retries: Optional[int] = None,
                         json_response: bool = False,
                         extract_pattern: Optional[str] = None) -> GenerationResponse:
        """
        Generate content using the configured models with retry and fallback logic.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Override the default max retries
            json_response: Whether to expect and parse a JSON response
            extract_pattern: Regex pattern to extract content from the response
            
        Returns:
            GenerationResponse object containing the generated text and metadata
        """
        max_retries = max_retries if max_retries is not None else self.default_max_retries
        
        # Try primary model first
        response = self._try_generate_with_provider(
            self.primary_config, prompt, max_retries, json_response, extract_pattern
        )
        
        if response:
            return response
        
        # Try fallback models in order
        for i, fallback_config in enumerate(self.fallback_configs):
            logger.info(f"Trying fallback model {i+1}: {fallback_config.provider.name} - {fallback_config.model_name}")
            
            response = self._try_generate_with_provider(
                fallback_config, prompt, max_retries, json_response, extract_pattern, is_fallback=True
            )
            
            if response:
                return response
        
        # If all models fail, raise an exception
        raise RuntimeError("All models failed to generate content after retries and fallbacks")
    
    def _try_generate_with_provider(self,
                                   config: ModelConfig,
                                   prompt: str,
                                   max_retries: int,
                                   json_response: bool,
                                   extract_pattern: Optional[str],
                                   is_fallback: bool = False) -> Optional[GenerationResponse]:
        """Try to generate content with a specific provider with retries"""
        retries = 0
        
        while retries < max_retries:
            try:
                # Check if we should rate limit
                if self._should_rate_limit(config.provider):
                    logger.info(f"Rate limiting {config.provider.name}, waiting...")
                    time.sleep(10)  # Simple backoff
                
                # Generate content based on provider
                if config.provider == ModelProvider.GEMINI:
                    return self._generate_with_gemini(config, prompt, json_response, extract_pattern, is_fallback)
                
                elif config.provider == ModelProvider.GROQ:
                    return self._generate_with_groq(config, prompt, json_response, extract_pattern, is_fallback)
                
                elif config.provider == ModelProvider.LMSTUDIO:
                    return self._generate_with_lmstudio(config, prompt, json_response, extract_pattern, is_fallback)
                
                # elif config.provider == ModelProvider.ANTHROPIC:
                #     return self._generate_with_anthropic(config, prompt, json_response, extract_pattern, is_fallback)
                
                else:
                    logger.error(f"Unsupported provider: {config.provider}")
                    return None
                    
            except ResourceExhausted as e:
                logger.warning(f"Rate limit hit with {config.provider.name} API: {str(e)}")
                retries += 1
                
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached for {config.provider.name}")
                    return None
                
                # Exponential backoff
                delay = self.initial_retry_delay * (2 ** (retries - 1))
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error generating content with {config.provider.name}: {str(e)}")
                return None
        
        return None
    
    def _generate_with_gemini(self,
                             config: ModelConfig, 
                             prompt: str,
                             json_response: bool,
                             extract_pattern: Optional[str],
                             is_fallback: bool) -> GenerationResponse:
        """Generate content using Gemini API"""
        model = genai.GenerativeModel(config.model_name)
        
        generation_config = {}
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.top_k is not None:
            generation_config["top_k"] = config.top_k
        if config.top_p is not None:
            generation_config["top_p"] = config.top_p
        if config.max_tokens is not None:
            generation_config["max_output_tokens"] = config.max_tokens
        
        raw_response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        self._update_usage_stats(ModelProvider.GEMINI)
        
        # Get response text
        response_text = raw_response.text
        
        # Process response if needed
        if json_response:
            response_text = self._extract_json(response_text)
        elif extract_pattern:
            response_text = self._extract_content(response_text, extract_pattern)
        
        return GenerationResponse(
            text=response_text,
            model_used=config.model_name,
            provider_used=config.provider,
            is_fallback=is_fallback,
            raw_response=raw_response
        )
    
    def _generate_with_groq(self,
                           config: ModelConfig, 
                           prompt: str,
                           json_response: bool,
                           extract_pattern: Optional[str],
                           is_fallback: bool) -> GenerationResponse:
        """Generate content using Groq API"""
        client = self.clients[ModelProvider.GROQ]["client"]
        
        # Convert parameters for Groq API
        chat_params = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        }
        
        if config.max_tokens is not None:
            chat_params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            chat_params["top_p"] = config.top_p
            
        raw_response = client.chat.completions.create(**chat_params)
        
        self._update_usage_stats(ModelProvider.GROQ)
        
        # Get response text
        response_text = raw_response.choices[0].message.content
        
        # Process response if needed
        if json_response:
            response_text = self._extract_json(response_text)
        elif extract_pattern:
            response_text = self._extract_content(response_text, extract_pattern)
        
        # Create usage dict from Groq response
        usage = {}
        if hasattr(raw_response, 'usage'):
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }
        
        return GenerationResponse(
            text=response_text,
            model_used=config.model_name,
            provider_used=config.provider,
            is_fallback=is_fallback,
            usage=usage,
            raw_response=raw_response
        )
    
    def _generate_with_lmstudio(self,
                               config: ModelConfig, 
                               prompt: str,
                               json_response: bool,
                               extract_pattern: Optional[str],
                               is_fallback: bool) -> GenerationResponse:
        """Generate content using LM Studio API (OpenAI-compatible)"""
        base_url = self.clients[ModelProvider.LMSTUDIO]["base_url"]
        api_key = self.clients[ModelProvider.LMSTUDIO]["api_key"]
        
        # LM Studio follows OpenAI API format
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key to headers if provided
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Prepare request payload
        payload = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        }
        
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        
        # Make the API request
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.request_timeout
        )
        
        # Raise exception if request failed
        response.raise_for_status()
        
        # Parse the response
        raw_response = response.json()
        
        self._update_usage_stats(ModelProvider.LMSTUDIO)
        
        # Extract the generated text from the response
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            response_text = raw_response["choices"][0]["message"]["content"]
        else:
            logger.error("Unexpected response format from LM Studio API")
            response_text = str(raw_response)
        
        # Process response if needed
        if json_response:
            response_text = self._extract_json(response_text)
        elif extract_pattern:
            response_text = self._extract_content(response_text, extract_pattern)
        
        # Create usage dict from response if available
        usage = {}
        if "usage" in raw_response:
            usage = {
                "prompt_tokens": raw_response["usage"].get("prompt_tokens", 0),
                "completion_tokens": raw_response["usage"].get("completion_tokens", 0),
                "total_tokens": raw_response["usage"].get("total_tokens", 0)
            }
        
        return GenerationResponse(
            text=response_text,
            model_used=config.model_name,
            provider_used=config.provider,
            is_fallback=is_fallback,
            usage=usage,
            raw_response=raw_response
        )
    
    # def _generate_with_anthropic(self,
    #                             config: ModelConfig, 
    #                             prompt: str,
    #                             json_response: bool,
    #                             extract_pattern: Optional[str],
    #                             is_fallback: bool) -> GenerationResponse:
    #     """Generate content using Anthropic API"""
    #     client = self.clients[ModelProvider.ANTHROPIC]["client"]
        
    #     message_params = {
    #         "model": config.model_name,
    #         "messages": [{"role": "user", "content": prompt}],
    #         "temperature": config.temperature,
    #     }
        
    #     if config.max_tokens is not None:
    #         message_params["max_tokens"] = config.max_tokens
    #     if config.top_p is not None:
    #         message_params["top_p"] = config.top_p
    #     if config.stop_sequences:
    #         message_params["stop_sequences"] = config.stop_sequences
            
    #     raw_response = client.messages.create(**message_params)
        
    #     self._update_usage_stats(ModelProvider.ANTHROPIC)
        
    #     # Get response text
    #     response_text = raw_response.content[0].text
        
    #     # Process response if needed
    #     if json_response:
    #         response_text = self._extract_json(response_text)
    #     elif extract_pattern:
    #         response_text = self._extract_content(response_text, extract_pattern)
        
    #     # Create usage dict from Anthropic response
    #     usage = {}
    #     if hasattr(raw_response, 'usage'):
    #         usage = {
    #             "input_tokens": raw_response.usage.input_tokens,
    #             "output_tokens": raw_response.usage.output_tokens
    #         }
        
    #     return GenerationResponse(
    #         text=response_text,
    #         model_used=config.model_name,
    #         provider_used=config.provider,
    #         is_fallback=is_fallback,
    #         usage=usage,
    #         raw_response=raw_response
    #     )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response"""
        try:
            # Try to find JSON pattern in the text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            # If no JSON pattern found, try to parse the entire text
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return original text if JSON parsing fails
            return {"error": "Failed to parse JSON", "text": text}
    
    def _extract_content(self, text: str, pattern: str) -> str:
        """Extract content using regex pattern"""
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        return text