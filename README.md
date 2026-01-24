# AI Infra

独立的 AI 基础设施模块，提供统一的 AI 模型接口、多供应商故障转移（Failover）、透明元数据封装以及流式传输支持。

## 功能特性

- **多供应商故障转移 (Failover)**：基于 YAML 配置，自动在多个提供商之间切换，确保高可用性。
- **透明元数据封装**：响应对象（`AIResponse`）完美兼容 `str`，但携带 `.model`、`.provider` 和 `.success` 等元数据。
- **双模式流式传输**：支持标准阻塞调用和异步迭代调用，且流式对象（`AIStream`）同样携带元数据。
- **配置驱动**：通过 `ai_models.yaml` 灵活定义逻辑模型到物理 Provider 的映射。
- **内置安全机制**：包含敏感信息掩码、重试逻辑、异常处理和详细日志记录。

## 安装

```bash
pip install -r requirements.txt
```

## 使用示例

### 1. 基础调用与元数据提取

```python
from ai_infra import init_ai_config, chat_completion

# 1. 根据逻辑模型名初始化配置列表（Failover 顺序）
configs = init_ai_config("gpt-oss")

# 2. 执行调用（内部自动处理 Failover）
response = await chat_completion(
    question="如何写一个高效的 Prompt？",
    configs=configs
)

# 3. 响应对象可以像普通字符串一样使用
print(f"Content: {response.strip()}")

# 4. 同时可以提取背后的供应商信息
print(f"Used Model: {response.model}")      # 实际响应的模型名
print(f"Provider: {response.provider}")    # 实际生效的提供商 ID
print(f"Success: {response.success}")      # 接口状态
```

### 2. 流式响应 (Streaming)

```python
configs = init_ai_config("gpt-oss")
stream = await chat_completion(
    question="讲一个长故事",
    configs=configs,
    streaming=True
)

async for chunk in stream:
    print(chunk, end="", flush=True)

# 迭代结束后，依然可以从 stream 对象获取元数据
print(f"\nFinal Model: {stream.model}")
```

## 配置说明 (`ai_models.yaml`)

在本项目的根目录下编辑 `ai_models.yaml`。逻辑名（如 `gpt-oss`）下定义的 `providers` 列表即为 Failover 顺序：

```yaml
models:
  gpt-oss:
    providers:
      - provider: AZURE
        model: gpt-4o
      - provider: OPENAI
        model: gpt-4o
      - provider: FREE_SERVER
        model: gpt-oss-120b
```

## 开发与扩展

- **新增供应商**：在 `ai_infra.py` 的 `_chat_completion` 函数中扩展 API 逻辑。
- **测试**：运行 `pytest` 目录下的测试用例。

## 依赖

- `openai` (v1.0+)
- `google-genai`
- `pyyaml`
- `tiktoken`
- `aiohttp`