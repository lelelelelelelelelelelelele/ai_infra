"""
流式响应功能示例
"""
import asyncio
from ai_infra import chat_completion, init_ai_config

async def streaming_example():
    """流式响应示例"""
    print("流式响应示例:")
    
    # 初始化配置
    config = init_ai_config("gpt")
    
    # 使用流式响应
    print("正在获取流式响应...")
    async for chunk in chat_completion(
        question="请简要介绍AI基础设施的重要性，分三个要点说明。",
        model=config[0]["model"] if isinstance(config, list) and config else config.get("model", "gpt-3.5-turbo"),
        base_url=config[0]["url"] if isinstance(config, list) and config else config.get("url", ""),
        api_key=config[0]["api_key"] if isinstance(config, list) and config else config.get("api_key", ""),
        streaming=True
    ):
        print(chunk, end="", flush=True)
    
    print("\n流式响应完成!")

async def non_streaming_example():
    """非流式响应示例（对比）"""
    print("\n非流式响应示例:")
    
    # 初始化配置
    config = init_ai_config("gpt")
    
    # 使用非流式响应
    response = await chat_completion(
        question="请简要介绍AI基础设施的重要性，分三个要点说明。",
        model=config[0]["model"] if isinstance(config, list) and config else config.get("model", "gpt-3.5-turbo"),
        base_url=config[0]["url"] if isinstance(config, list) and config else config.get("url", ""),
        api_key=config[0]["api_key"] if isinstance(config, list) and config else config.get("api_key", ""),
        streaming=False
    )
    
    print(f"完整响应: {response}")

if __name__ == "__main__":
    asyncio.run(streaming_example())
    asyncio.run(non_streaming_example())