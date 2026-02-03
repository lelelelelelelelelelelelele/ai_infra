"""High-level AI Infra integration smoke tests.

These tests focus on config loading and basic wiring. Network calls are skipped if no API key exists.
"""

import pytest

from ai_infra.ai_infra import (
	ai_infra,
	chat_completion,
	get_ai_models,
	init_ai_config,
	load_secrets_from_env,
)


def test_ai_config_smoke():
	load_secrets_from_env()

	models = get_ai_models()
	assert isinstance(models, list)
	assert models

	default_configs = init_ai_config("default")
	assert isinstance(default_configs, list)
	assert default_configs

	gemini_configs = init_ai_config("gemini")
	assert isinstance(gemini_configs, list)
	assert gemini_configs


@pytest.mark.asyncio
async def test_chat_completion_smoke():
	load_secrets_from_env()

	configs = init_ai_config("default")
	if not configs or not configs[0].get("api_key"):
		pytest.skip("API key not configured")

	response = await chat_completion(
		question="Reply with 'OK' only.",
		configs=configs,
	)
	assert str(response).strip() == "OK"


def test_ai_infra_instance_smoke():
	# Legacy compatibility: global AIInfra instance exists.
	assert ai_infra is not None
	assert hasattr(ai_infra, "models")
