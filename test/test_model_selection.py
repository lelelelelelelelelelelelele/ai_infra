"""Model selection and config resolution tests (AI Infra)."""

import pytest

from ai_infra.ai_infra import get_ai_models, init_ai_config, load_secrets_from_env


@pytest.mark.asyncio
async def test_model_apis():
	load_secrets_from_env()

	models = get_ai_models()
	assert isinstance(models, list)
	assert models, "No models found"

	# Verify config resolution returns a provider list.
	test_models = ["gpt", "gemini", "deepseek", "kimi", "default"]
	for model in test_models:
		configs = init_ai_config(model)
		assert isinstance(configs, list)
		assert configs, f"No configs for model '{model}'"
		_ = bool(configs[0].get("api_key"))
