# AI Infra - AI API通信基础设施
# 提供统一的AI模型接口、缓存、监控和错误处理

import asyncio
import logging
import yaml
import os
import warnings
from typing import Dict, Any, List, Optional, Union
import io

# OpenAI SDK imports
try:
    from openai import OpenAI
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
# AI 响应分装类 (AI Response Wrappers)
# ================================

class AIResponse(str):
    """
    包装后的字符串响应，支持直接作为字符串使用，
    同时也携带 model, success 等元数据。
    """
    def __new__(cls, content, model, success=True, provider=None):
        obj = str.__new__(cls, content)
        obj.model = model
        obj.success = success
        obj.provider = provider
        return obj

class AIStream:
    """
    包装后的流式响应，代理异步生成器，
    同时也携带 model, success 等元数据。
    """
    def __init__(self, generator, model, success=True, provider=None):
        self._generator = generator
        self.model = model
        self.success = success
        self.provider = provider

    def __aiter__(self):
        return self._generator.__aiter__()

# ================================
# 融合的用户提供的AI功能函数
# ================================

async def _chat_completion(question: str, model: str, base_url: str, api_key: str, system_instr: str | None = None, streaming: bool = False, provider: str | None = None):
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
    provider : str, optional
        The provider name for metadata.
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
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    choice0 = choices[0]
                    delta = getattr(choice0, "delta", None)
                    if isinstance(delta, dict):
                        content = delta.get("content")
                    else:
                        content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        full_content += content
                        yield content
                logger.info(f"Model: {model}, Base URL: {base_url}\nStream response: {full_content}\n")
            
            # 返回包装后的流对象
            return AIStream(stream_generator(), model=model, provider=provider)
            
        except Exception as e:
            # 尝试 2: 降级逻辑
            if system_instr is not None:
                logger.warning(f"Chat completion failed with system role. Retrying by merging system instruction into user prompt. Error: {e}")
                
                merged_content = f"{system_instr}\n\n{question}"
                messages = [{"role": "user", "content": merged_content}]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages, # type: ignore
                    stream=True
                )
                
                async def stream_generator():
                    full_content = ""
                    for chunk in response:
                        choices = getattr(chunk, "choices", None)
                        if not choices:
                            continue
                        choice0 = choices[0]
                        delta = getattr(choice0, "delta", None)
                        if isinstance(delta, dict):
                            content = delta.get("content")
                        else:
                            content = getattr(delta, "content", None) if delta is not None else None
                        if content:
                            full_content += content
                            yield content
                    logger.info(f"Model: {model}, Base URL: {base_url}\nStream response: {full_content}\n")
                
                return AIStream(stream_generator(), model=model, provider=provider)
            
            raise e
    else:
        # 非流式响应模式
        def _sync_call():
            client = OpenAI(api_key=api_key, base_url=base_url)
            
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
                if system_instr is not None:
                    logger.warning(f"Chat completion failed with system role. Retrying... Error: {e}")
                    merged_content = f"{system_instr}\n\n{question}"
                    messages = [{"role": "user", "content": merged_content}]
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages # type: ignore
                    )
                    return response.choices[0].message.content or ""
                raise e

        content = await asyncio.get_event_loop().run_in_executor(None, _sync_call)
        logger.info(f"Model: {model}, Base URL: {base_url}\nResponse: {content}\n")
        # 返回包装后的字符串对象
        return AIResponse(content, model=model, provider=provider)

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
    os.path.join(os.path.dirname(__file__), "..", "ai_infra", "ai_models.yaml")
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
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, OSError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return True
    return False

def _init_ai_config_fallback(name: str) -> dict[str, str]:
    warnings.warn(
        "This method is deprecated, use YAML config instead", 
        DeprecationWarning, 
        stacklevel=2
    )
    free_api_url = "https://wherexianyi.zeabur.app/v1"  # was lastxianyi (502)
    chat_api_url = "https://api.chatanywhere.org/v1"

    if "default" in name or name == "gpt-oss":
        config = {
            "url": chat_api_url,
            "model": "gpt-5-mini",
        }
        api_model = "CHAT"
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
            "model": "qwen/qwen3.5-397b-a17b",
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
            "url": free_api_url,
            "model": "deepseek-ai/deepseek-v4-pro",
        }
        api_model = "FREE"
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
    model_name: str = "",
    system_instr: str | None = None,
    configs: Optional[List[Dict[str, str]]] = None,
    streaming: bool = False,
) -> Union[str, Any]:
    if model_name == "" and configs is None:
        raise ValueError("Either model_name or configs must be provided.")

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
                streaming=streaming,
                provider=config.get("provider")
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
load_secrets_from_env(".env")
