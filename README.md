# AI Infra

独立的AI基础设施模块，提供统一的AI模型接口、配置管理和API通信。

## 功能特性

- 支持多种AI模型（OpenAI、Gemini、DeepSeek等）
- YAML配置驱动，支持灵活的模型映射
- 异步API调用
- 错误处理和日志记录
- PDF文件交互支持

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```python
from ai_infra import init_ai_config, chat_completion_with_failover

# 初始化配置（返回按优先顺序的 provider 列表）
configs = init_ai_config("qwen-max")
print(configs)

# 进行聊天完成（失败自动切换到下一个 provider）
response = await chat_completion_with_failover(
    question="Hello, world!",
    model_name="qwen-max",
	configs=configs
)
```

## 配置

编辑 `config/ai_models.yaml` 来添加或修改模型配置。

## 依赖

- openai
- google-genai
- pyyaml
- python-dotenv
- aiohttp
- redis
- tiktoken