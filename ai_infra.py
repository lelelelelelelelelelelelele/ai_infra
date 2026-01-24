# AI Infra - AI API通信基础设施
# 提供统一的AI模型接口、缓存、监控和错误处理

import asyncio
import logging
import time
import json
import yaml
import hashlib
import os
import warnings
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import aiohttp
import redis
from datetime import datetime, timedelta
import tiktoken
import io

# OpenAI SDK imports
try:
    from openai import OpenAI
    from openai.types.chat import (
        ChatCompletionUserMessageParam,
        ChatCompletionSystemMessageParam,
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Google Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# dotenv imports
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)

# ================================
# 融合的用户提供的AI功能函数
# ================================

async def _chat_completion(question: str, model: str, base_url: str, api_key: str, system_instr: str | None = None, streaming: bool = False):
    '''
    AI interaction function using OpenAI SDK.
    Parameters
    ----------
    question : str
        The input question or prompt to send to the AI model.
    model : str
        The AI model to use for generating the response.
    base_url : str
        The base URL of the AI service API.
    api_key : str
        API key for authentication.
    system_instr : str, optional
        System instruction for the AI model, by default None.
    streaming : bool, optional
        Whether to return streaming response, by default False.
    '''
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI SDK not available. Install with: pip install openai")

    if streaming:
        # 流式响应模式
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 尝试 1: 标准方式，使用 system role
        try:
            messages: List[Dict[str, str]] = []
            if system_instr is not None:
                messages.append({"role": "system", "content": system_instr})
            messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                stream=True
            )
            
            # 生成器函数，逐个返回响应内容
            async def stream_generator():
                full_content = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_content += content
                        yield content
                logger.info(f"Model: {model}, Base URL: {base_url}\nStream response: {full_content}\n")
                
            return stream_generator()
            
        except Exception as e:
            # 尝试 2: 如果使用了 system_instr 且调用失败 (如 400/422 不支持 system role)，
            # 则降级为将 system instruction 拼接到 user message 中
            if system_instr is not None:
                logger.warning(f"Chat completion failed with system role. Retrying by merging system instruction into user prompt. Error: {e}")
                
                merged_content = f"{system_instr}\n\n{question}"
                messages = [{"role": "user", "content": merged_content}]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages, # type: ignore
                    stream=True
                )
                
                # 生成器函数，逐个返回响应内容
                async def stream_generator():
                    full_content = ""
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_content += content
                            yield content
                    logger.info(f"Model: {model}, Base URL: {base_url}\nStream response: {full_content}\n")
                    
                return stream_generator()
            
            # 如果没有使用 system_instr 或者重试也失败，则抛出异常
            raise e
    else:
        # 非流式响应模式
        def _sync_call():
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            # 尝试 1: 标准方式，使用 system role
            try:
                messages: List[Dict[str, str]] = []
                if system_instr is not None:
                    messages.append({"role": "system", "content": system_instr})
                messages.append({"role": "user", "content": question})
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages # type: ignore
                )
                return response.choices[0].message.content or ""
                
            except Exception as e:
                # 尝试 2: 如果使用了 system_instr 且调用失败 (如 400/422 不支持 system role)，
                # 则降级为将 system instruction 拼接到 user message 中
                if system_instr is not None:
                    logger.warning(f"Chat completion failed with system role. Retrying by merging system instruction into user prompt. Error: {e}")
                    
                    merged_content = f"{system_instr}\n\n{question}"
                    messages = [{"role": "user", "content": merged_content}]
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages # type: ignore
                    )
                    return response.choices[0].message.content or ""
                
                # 如果没有使用 system_instr 或者重试也失败，则抛出异常
                raise e

        content = await asyncio.get_event_loop().run_in_executor(None, _sync_call)
        logger.info(f"Model: {model}, Base URL: {base_url}\nResponse: {content}\n")
        return content

def init_genai_client(api_key: str | None = None) -> 'genai.Client':
    """Initialize Gemini AI client"""
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI SDK not available. Install with: pip install google-generativeai")

    if not api_key:
        api_key = os.environ.get("GEMINI_KEY") or ""
    if not api_key:
        raise ValueError("API key for Gemini is not provided.")
    return genai.Client(api_key=api_key)

def interact_with_pdf(client: 'genai.Client', file: Union[str, os.PathLike[str], io.IOBase], question: str = "") -> str:
    '''
    AI interaction function using Generative SDK to interact with PDF files.
    Parameters
    ----------
    client : genai.Client
        Initialized Gemini client
    file :
        A path to the file or an `IOBase` object to be uploaded. If it's an
        IOBase object, it must be opened in blocking (the default) mode and
        binary mode.
    question : str
        The input question or prompt to send to the AI model.
    '''
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI SDK not available")

    # 上传文件
    pdf_file = client.files.upload(file=file)
    # 生成内容
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            question,
            pdf_file
        ]
    )
    content = response.text or ""
    logger.info(f"PDF interaction response: {content}")
    return content

MODEL_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "ai_models.yaml")
)


def load_model_config(config_path: str | None = None) -> Dict[str, Any]:
    path = os.path.normpath(config_path or MODEL_CONFIG_PATH)
    if not os.path.exists(path):
        logger.warning(f"Model config YAML not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error(f"Failed to load model config YAML: {exc}")
        return {}


def _resolve_model_entry(name: str, config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    models = config_data.get("models", {})
    if not isinstance(models, dict):
        return None
    if name in models:
        return models.get(name)
    for entry in models.values():
        matches = entry.get("match", []) if isinstance(entry, dict) else []
        if isinstance(matches, list) and any(m in name for m in matches):
            return entry
    return None


def _build_config_from_yaml(name: str, config_data: Dict[str, Any]) -> List[Dict[str, str]]:
    entry = _resolve_model_entry(name, config_data)
    if not entry:
        default_key = config_data.get("default_model")
        if isinstance(default_key, str):
            entry = config_data.get("models", {}).get(default_key)
    if not entry:
        return []

    provider_entries = entry.get("providers", [])
    if not isinstance(provider_entries, list):
        return []

    providers = config_data.get("providers", {})
    if not isinstance(providers, dict):
        return []

    configs: List[Dict[str, str]] = []
    for provider_entry in provider_entries:
        if not isinstance(provider_entry, dict):
            continue

        provider_key = provider_entry.get("provider")
        if not provider_key:
            continue

        provider = providers.get(provider_key, {})
        if not isinstance(provider, dict):
            continue

        base_url = provider_entry.get("base_url") or provider.get("base_url", "")
        api_key_env = provider_entry.get("api_key_env") or provider.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        configs.append({
            "url": base_url,
            "model": provider_entry.get("model", ""),
            "api_key": api_key,
            "provider": provider_key,
        })

    return configs


def _is_network_or_http_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, aiohttp.ClientError, OSError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return True
    return False

import warnings

def _init_ai_config_fallback(name: str) -> dict[str, str]:
    warnings.warn(
        "This method is deprecated, use YAML config instead", 
        DeprecationWarning, 
        stacklevel=2
    )
    free_api_url = "https://lastxianyi.zeabur.app/v1"
    chat_api_url = "https://api.chatanywhere.org/v1"

    if "default" in name or name == "gpt-oss":
        config = {
            "url": free_api_url,
            "model": "openai/gpt-oss-120b",
        }
        api_model = "FREE"
    elif "st" in name or "spark" in name:
        config = {
            "url": "https://spark-api-open.xf-yun.com/v1",
            "model": "lite",
        }
        api_model = "ST"
    elif "gemini" in name or "google" in name:
        config = {
            "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.5-flash",
        }
        api_model = "GEMINI"
    elif "kimi" in name or "moonshot" in name:
        config = {
            "url": free_api_url,
            "model": "moonshotai/Kimi-K2-Instruct",
        }
        api_model = "FREE"
    elif "qwen" in name:
        config = {
            "url": free_api_url,
            "model": "tongyi-qwen3-max-model",
        }
        api_model = "FREE"
    elif "glm" in name:
        config = {
            "url": free_api_url,
            "model": "zai-org/GLM-4.5",
        }
        api_model = "FREE"
    elif "gpt" in name or "chatgpt" in name:
        config = {
            "url": chat_api_url,
            "model": "gpt-5-mini",
        }
        api_model = "CHAT"
    elif name == "gpt-4o-mini":
        config = {
            "url": chat_api_url,
            "model": "gpt-4o-mini",
        }
        api_model = "CHAT"
    elif "deepseek" in name or "ds" in name:
        config = {
            "url": chat_api_url,
            "model": "deepseek-v3",
        }
        api_model = "CHAT"
    else:
        config = {
            "url": chat_api_url,
            "model": "gpt-3.5-turbo",
        }
        api_model = "GPT"

    config["api_key"] = os.environ.get(f"{api_model}_KEY") or ""
    return config


def init_ai_config(model: str = "default") -> List[Dict[str, str]]:
    """
    Initialize configuration for different AI models based on the model name.
    Uses YAML config if available, otherwise falls back to static mapping.
    """
    name = model.lower()
    config_data = load_model_config()
    if config_data:
        yaml_configs = _build_config_from_yaml(name, config_data)
        if yaml_configs:
            return yaml_configs
    else:
        return [_init_ai_config_fallback(name)]


async def chat_completion(
    question: str,
    model_name: str,
    system_instr: str | None = None,
    configs: Optional[List[Dict[str, str]]] = None,
    streaming: bool = False,
) -> Union[str, Any]:
    configs = configs or init_ai_config(model_name)
    last_error: Optional[Exception] = None

    for config in configs:
        api_key = config.get("api_key", "")
        if not api_key:
            continue

        try:
            return await _chat_completion(
                question=question,
                model=config.get("model", ""),
                base_url=config.get("url", ""),
                api_key=api_key,
                system_instr=system_instr,
                streaming=streaming
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_network_or_http_error(exc):
                logger.warning(
                    "Provider failed, trying next. provider=%s model=%s error=%s",
                    config.get("provider", ""),
                    config.get("model", ""),
                    exc,
                )
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("No available provider configuration found.")


def get_ai_models() -> list:
    """
    返回所有在 init_ai_config 中定义的 AI 模型名称列表。
    default: gpt
    """
    config_data = load_model_config()
    models = config_data.get("models") if isinstance(config_data, dict) else None
    if isinstance(models, dict) and models:
        return list(models.keys())
    return ["gpt", "gemini", "deepseek", "kimi", "gpt-oss", "spark", "qwen", "glm"]

def load_secrets_from_env(env_file: str = "secret.env"):
    """Load secrets from environment file"""
    if DOTENV_AVAILABLE and os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loaded secrets from {env_file}")
    else:
        logger.warning(f"Could not load secrets from {env_file}. dotenv not available or file not found.")

# 初始化时加载环境变量
load_secrets_from_env()


# class ModelProvider(Enum):
#     """AI模型提供商"""
#     OPENAI = "openai"
#     ANTHROPIC = "anthropic"
#     GOOGLE = "google"
#     LOCAL = "local"

# @dataclass
# class ModelConfig:
#     """模型配置"""
#     name: str
#     provider: str
#     api_key: str
#     base_url: Optional[str] = None
#     max_tokens: int = 2048
#     temperature: float = 0.7
#     top_p: float = 1.0
#     timeout: int = 30
#     retry_count: int = 3
#     retry_delay: float = 1.0

# @dataclass
# class APIConfig:
#     """API配置"""
#     rate_limit: int = 100  # 每分钟请求数
#     batch_size: int = 10
#     enable_cache: bool = True
#     enable_monitoring: bool = True
#     enable_fallback: bool = True

# @dataclass
# class ModelResponse:
#     """模型响应"""
#     content: str
#     model: str
#     usage: Dict[str, int]
#     latency: float
#     cached: bool = False
#     error: Optional[str] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)

# class BaseModel(ABC):
#     """基础模型类"""
    
#     def __init__(self, config: ModelConfig):
#         self.config = config
#         self.name = config.name
#         self.provider = config.provider
#         self._session: Optional[aiohttp.ClientSession] = None
    
#     @abstractmethod
#     async def generate(self, prompt: str, **kwargs) -> ModelResponse:
#         """生成文本"""
#         pass
    
#     @abstractmethod
#     async def generate_stream(self, prompt: str, **kwargs):
#         """流式生成文本"""
#         pass
    
#     @abstractmethod
#     def count_tokens(self, text: str) -> int:
#         """计算token数量"""
#         pass
    
#     async def __aenter__(self):
#         self._session = aiohttp.ClientSession()
#         return self
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         if self._session:
#             await self._session.close()

# class OpenAIModel(BaseModel):
#     """OpenAI模型实现"""
    
#     def __init__(self, config: ModelConfig):
#         super().__init__(config)
#         self.base_url = config.base_url or "https://api.openai.com/v1"
#         self.headers = {
#             "Authorization": f"Bearer {config.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         # 初始化tokenizer
#         try:
#             self.tokenizer = tiktoken.encoding_for_model(config.name)
#         except:
#             self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
#     async def generate(self, prompt: str, **kwargs) -> ModelResponse:
#         """生成文本"""
#         start_time = time.time()
        
#         try:
#             # 构建请求数据
#             data = {
#                 "model": self.name,
#                 "messages": [
#                     {"role": "user", "content": prompt}
#                 ],
#                 "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
#                 "temperature": kwargs.get("temperature", self.config.temperature),
#                 "top_p": kwargs.get("top_p", self.config.top_p)
#             }
            
#             # 添加系统Prompt
#             if "system_prompt" in kwargs:
#                 data["messages"].insert(0, {
#                     "role": "system",
#                     "content": kwargs["system_prompt"]
#                 })
            
#             # 发送请求
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     f"{self.base_url}/chat/completions",
#                     headers=self.headers,
#                     json=data,
#                     timeout=aiohttp.ClientTimeout(total=self.config.timeout)
#                 ) as response:
                    
#                     if response.status == 200:
#                         result = await response.json()
#                         content = result["choices"][0]["message"]["content"]
#                         usage = result["usage"]
                        
#                         latency = time.time() - start_time
                        
#                         return ModelResponse(
#                             content=content,
#                             model=self.name,
#                             usage=usage,
#                             latency=latency,
#                             metadata={
#                                 "finish_reason": result["choices"][0]["finish_reason"],
#                                 "response_id": result["id"]
#                             }
#                         )
#                     else:
#                         error_text = await response.text()
#                         raise Exception(f"OpenAI API错误: {response.status} - {error_text}")
        
#         except Exception as e:
#             logger.error(f"OpenAI模型生成错误: {e}")
#             return ModelResponse(
#                 content="",
#                 model=self.name,
#                 usage={},
#                 latency=time.time() - start_time,
#                 error=str(e)
#             )
    
#     async def generate_stream(self, prompt: str, **kwargs):
#         """流式生成文本"""
#         data = {
#             "model": self.name,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
#             "temperature": kwargs.get("temperature", self.config.temperature),
#             "stream": True
#         }
        
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{self.base_url}/chat/completions",
#                 headers=self.headers,
#                 json=data
#             ) as response:
#                 async for line in response.content:
#                     if line:
#                         line = line.decode('utf-8').strip()
#                         if line.startswith('data: '):
#                             data_str = line[6:]
#                             if data_str != '[DONE]':
#                                 try:
#                                     data = json.loads(data_str)
#                                     if "choices" in data and data["choices"]:
#                                         delta = data["choices"][0]["delta"]
#                                         if "content" in delta:
#                                             yield delta["content"]
#                                 except:
#                                     continue
    
#     def count_tokens(self, text: str) -> int:
#         """计算token数量"""
#         try:
#             return len(self.tokenizer.encode(text))
#         except:
#             return int(len(text.split()) * 1.3)  # 粗略估计

# class AnthropicModel(BaseModel):
#     """Anthropic模型实现"""
    
#     def __init__(self, config: ModelConfig):
#         super().__init__(config)
#         self.base_url = config.base_url or "https://api.anthropic.com/v1"
#         self.headers = {
#             "x-api-key": config.api_key,
#             "Content-Type": "application/json",
#             "anthropic-version": "2023-06-01"
#         }
    
#     async def generate(self, prompt: str, **kwargs) -> ModelResponse:
#         """生成文本"""
#         start_time = time.time()
        
#         try:
#             data = {
#                 "model": self.name,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
#                 "temperature": kwargs.get("temperature", self.config.temperature)
#             }
            
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     f"{self.base_url}/messages",
#                     headers=self.headers,
#                     json=data,
#                     timeout=aiohttp.ClientTimeout(total=self.config.timeout)
#                 ) as response:
                    
#                     if response.status == 200:
#                         result = await response.json()
#                         content = result["content"][0]["text"]
                        
#                         latency = time.time() - start_time
                        
#                         return ModelResponse(
#                             content=content,
#                             model=self.name,
#                             usage=result.get("usage", {}),
#                             latency=latency,
#                             metadata={
#                                 "response_id": result.get("id", "")
#                             }
#                         )
#                     else:
#                         error_text = await response.text()
#                         raise Exception(f"Anthropic API错误: {response.status} - {error_text}")
        
#         except Exception as e:
#             logger.error(f"Anthropic模型生成错误: {e}")
#             return ModelResponse(
#                 content="",
#                 model=self.name,
#                 usage={},
#                 latency=time.time() - start_time,
#                 error=str(e)
#             )
    
#     async def generate_stream(self, prompt: str, **kwargs):
#         """流式生成文本"""
#         # Anthropic的流式API实现
#         data = {
#             "model": self.name,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
#             "temperature": kwargs.get("temperature", self.config.temperature),
#             "stream": True
#         }
        
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{self.base_url}/messages",
#                 headers=self.headers,
#                 json=data
#             ) as response:
#                 async for line in response.content:
#                     if line:
#                         line = line.decode('utf-8').strip()
#                         if line.startswith('data: '):
#                             data_str = line[6:]
#                             try:
#                                 data = json.loads(data_str)
#                                 if "delta" in data and "text" in data["delta"]:
#                                     yield data["delta"]["text"]
#                             except:
#                                 continue
    
#     def count_tokens(self, text: str) -> int:
#         """计算token数量"""
#         # Anthropic使用不同的token计算方式
#         return int(len(text.split()) * 1.2)  # 粗略估计

# class RedisCache:
#     """Redis缓存实现"""
    
#     def __init__(self, redis_url: str = "redis://localhost:6379"):
#         self.redis_client = redis.from_url(redis_url, decode_responses=True)
#         self.default_ttl = 3600  # 默认1小时
    
#     def _generate_key(self, prompt: str, model: str, **kwargs) -> str:
#         """生成缓存键"""
#         content = f"{prompt}_{model}_{json.dumps(kwargs, sort_keys=True)}"
#         return f"ai_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
#     async def get(self, key: str) -> Optional[str]:
#         """获取缓存"""
#         try:
#             return self.redis_client.get(key) # type: ignore
#         except Exception as e:
#             logger.error(f"Redis获取缓存错误: {e}")
#             return None
    
#     async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
#         """设置缓存"""
#         try:
#             ttl = ttl or self.default_ttl
#             return self.redis_client.setex(key, ttl, value) # type: ignore
#         except Exception as e:
#             logger.error(f"Redis设置缓存错误: {e}")
#             return False
    
#     async def generate_and_cache(self, key: str, generate_func, ttl: Optional[int] = None):
#         """生成并缓存"""
#         # 检查缓存
#         cached = await self.get(key)
#         if cached:
#             return cached, True
        
#         # 生成新内容
#         result = await generate_func()
        
#         # 缓存结果
#         if result and not result.get("error"):
#             await self.set(key, json.dumps(result), ttl)
        
#         return result, False

# class PerformanceMonitor:
#     """性能监控器"""
    
#     def __init__(self):
#         self.metrics = {
#             "total_requests": 0,
#             "successful_requests": 0,
#             "failed_requests": 0,
#             "average_latency": 0.0,
#             "token_usage": 0,
#             "error_rate": 0.0
#         }
#         self.request_times = []
#         self.errors = []
    
#     def record_request(self, latency: float, success: bool, tokens: int = 0):
#         """记录请求"""
#         self.metrics["total_requests"] += 1
#         self.request_times.append(latency)
        
#         if success:
#             self.metrics["successful_requests"] += 1
#         else:
#             self.metrics["failed_requests"] += 1
#             self.errors.append({"timestamp": datetime.now(), "latency": latency})
        
#         self.metrics["token_usage"] += tokens
        
#         # 更新平均延迟
#         if len(self.request_times) > 0:
#             self.metrics["average_latency"] = sum(self.request_times[-100:]) / len(self.request_times[-100:])
        
#         # 更新错误率
#         self.metrics["error_rate"] = self.metrics["failed_requests"] / self.metrics["total_requests"]
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """获取监控指标"""
#         return {
#             **self.metrics,
#             "recent_requests": len(self.request_times),
#             "recent_errors": len(self.errors),
#             "uptime": time.time() - getattr(self, "start_time", time.time())
#         }

# class CircuitBreaker:
#     """熔断器"""
    
#     def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
#         self.failure_threshold = failure_threshold
#         self.recovery_timeout = recovery_timeout
#         self.failure_count = 0
#         self.last_failure_time = None
#         self.state = "closed"  # closed, open, half-open
    
#     async def call(self, func, *args, **kwargs):
#         """调用函数，应用熔断逻辑"""
#         if self.state == "open":
#             if self.last_failure_time is None:
#                 self.state = "half-open"
#             elif time.time() - self.last_failure_time > self.recovery_timeout:
#                 self.state = "half-open"
#             else:
#                 raise Exception("Circuit breaker is open")
        
#         try:
#             result = await func(*args, **kwargs)
#             if self.state == "half-open":
#                 self.state = "closed"
#                 self.failure_count = 0
#             return result
#         except Exception as e:
#             self.failure_count += 1
#             self.last_failure_time = time.time()
            
#             if self.failure_count >= self.failure_threshold:
#                 self.state = "open"
            
#             raise e

# class AIInfra:
#     """AI基础设施主类"""
    
#     def __init__(self):
#         self.models: Dict[str, BaseModel] = {}
#         self.cache = RedisCache()
#         self.monitor = PerformanceMonitor()
#         self.circuit_breakers: Dict[str, CircuitBreaker] = {}
#         self.api_config = APIConfig()
    
#     def register_model(self, name: str, config: ModelConfig):
#         """注册模型"""
#         if config.provider == "openai":
#             self.models[name] = OpenAIModel(config)
#         elif config.provider == "anthropic":
#             self.models[name] = AnthropicModel(config)
#         else:
#             raise ValueError(f"不支持的提供商: {config.provider}")
        
#         # 为每个模型创建熔断器
#         self.circuit_breakers[name] = CircuitBreaker()
#         logger.info(f"模型 {name} 已注册")
    
#     async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
#         """生成文本"""
#         if model not in self.models:
#             raise ValueError(f"模型 {model} 未注册")
        
#         model_instance = self.models[model]
#         circuit_breaker = self.circuit_breakers[model]
        
#         # 生成缓存键
#         cache_key = self.cache._generate_key(prompt, model, **kwargs)
        
#         # 检查缓存
#         if self.api_config.enable_cache:
#             cached_result = await self.cache.get(cache_key)
#             if cached_result:
#                 return json.loads(cached_result)
        
#         start_time = time.time()
        
#         try:
#             # 使用熔断器调用模型
#             response = await circuit_breaker.call(
#                 model_instance.generate,
#                 prompt,
#                 **kwargs
#             )
            
#             latency = time.time() - start_time
            
#             # 记录监控数据
#             if self.api_config.enable_monitoring:
#                 self.monitor.record_request(
#                     latency=latency,
#                     success=response.error is None,
#                     tokens=response.usage.get("total_tokens", 0) if response.usage else 0
#                 )
            
#             # 缓存结果
#             if self.api_config.enable_cache and not response.error:
#                 result_data = {
#                     "content": response.content,
#                     "model": response.model,
#                     "usage": response.usage,
#                     "cached": True,
#                     "timestamp": datetime.now().isoformat()
#                 }
#                 await self.cache.set(cache_key, json.dumps(result_data))
            
#             return {
#                 "content": response.content,
#                 "model": response.model,
#                 "usage": response.usage,
#                 "latency": latency,
#                 "cached": False,
#                 "error": response.error
#             }
            
#         except Exception as e:
#             latency = time.time() - start_time
            
#             if self.api_config.enable_monitoring:
#                 self.monitor.record_request(latency, False)
            
#             logger.error(f"生成文本错误: {e}")
            
#             # 回退到备用模型
#             if self.api_config.enable_fallback:
#                 fallback_result = await self._try_fallback(model, prompt, **kwargs)
#                 if fallback_result:
#                     return fallback_result
            
#             return {
#                 "content": "",
#                 "error": str(e),
#                 "latency": latency
#             }
    
#     async def _try_fallback(self, original_model: str, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
#         """尝试使用备用模型"""
#         fallback_models = [name for name in self.models.keys() if name != original_model]
        
#         for fallback_model in fallback_models:
#             try:
#                 logger.info(f"尝试使用备用模型: {fallback_model}")
#                 return await self.generate(fallback_model, prompt, **kwargs)
#             except Exception as e:
#                 logger.warning(f"备用模型 {fallback_model} 失败: {e}")
#                 continue
        
#         return None
    
#     async def generate_batch(self, model: str, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
#         """批量生成文本"""
#         batch_size = self.api_config.batch_size
#         results = []
        
#         for i in range(0, len(prompts), batch_size):
#             batch = prompts[i:i + batch_size]
#             batch_tasks = [
#                 self.generate(model, prompt, **kwargs)
#                 for prompt in batch
#             ]
#             batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
#             results.extend(batch_results)
        
#         return results
    
#     def count_tokens(self, model: str, text: str) -> int:
#         """计算token数量"""
#         if model in self.models:
#             return self.models[model].count_tokens(text)
#         return int(len(text.split()) * 1.3)  # 默认估计
    
#     def get_model_info(self, model: str) -> Dict[str, Any]:
#         """获取模型信息"""
#         if model not in self.models:
#             return {}
        
#         return {
#             "name": model,
#             "provider": self.models[model].provider,
#             "config": self.models[model].config.__dict__,
#             "status": "active"
#         }
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """获取系统指标"""
#         return {
#             "monitoring": self.monitor.get_metrics(),
#             "registered_models": list(self.models.keys()),
#             "cache_status": "active" if self.api_config.enable_cache else "disabled",
#             "circuit_breakers": {
#                 name: breaker.state for name, breaker in self.circuit_breakers.items()
#             }
#         }

# # 全局AI Infra实例
# ai_infra = AIInfra()

