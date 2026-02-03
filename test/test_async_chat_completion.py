"""Async chat_completion smoke test (AI Infra).

Network/integration test: skipped if API key is missing.
"""

import pytest

from ai_infra.ai_infra import chat_completion, init_ai_config, load_secrets_from_env


@pytest.mark.asyncio
async def test_async_chat_completion_smoke():
	load_secrets_from_env()

	configs = init_ai_config("gpt")
	if not configs or not configs[0].get("api_key"):
		pytest.skip("No API key configured")

	response = await chat_completion(
		question="Say hello in Chinese",
		configs=configs,
	)
	assert response
