"""Pytest example for chat completion."""
import asyncio

import pytest

from ai_infra import chat_completion, init_ai_config


def test_chat_completion_gpt():
    configs = init_ai_config("gpt")
    print("Loaded configs:", configs)
    if not any(config.get("api_key") for config in configs):
        pytest.skip("Missing API key for GPT provider")

    response = asyncio.run(
        chat_completion(
            question="请简要介绍AI基础设施的重要性，分三个要点说明。",
            model_name="gpt",
        )
    )

    assert isinstance(response, str)
    assert response.strip()


def test_chat_completion_gpt_streaming():
    configs = init_ai_config("gpt")
    print("Loaded configs:", configs)
    if not any(config.get("api_key") for config in configs):
        pytest.skip("Missing API key for GPT provider")

    async def _run_stream() -> str:
        chunks = []
        async for chunk in await chat_completion(
            question="请简要介绍AI基础设施的重要性，分三个要点说明。",
            model_name="gpt",
            streaming=True,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    response = asyncio.run(_run_stream())
    assert isinstance(response, str)
    assert response.strip()