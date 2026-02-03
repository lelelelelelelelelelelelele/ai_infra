"""AI Infra connectivity tests.

These tests are network/integration tests. They will be skipped if no API key is configured.
"""

import asyncio

import pytest

from ai_infra.ai_infra import chat_completion, init_ai_config


@pytest.mark.asyncio
async def test_single_model_connectivity(model_name: str):
	"""Test connectivity for a single logical model name."""
	print(f"\nTesting model: {model_name} ...", end=" ", flush=True)

	configs = init_ai_config(model_name)
	if not configs:
		pytest.skip("No providers configured for this model")

	if not configs[0].get("api_key"):
		pytest.skip("No API Key configured")

	start_time = asyncio.get_event_loop().time()
	try:
		response = await chat_completion(
			question="Ping. Reply with 'Pong' only.",
			configs=configs,
		)
	except Exception as exc:  # pragma: no cover - network/account surface
		message = str(exc)
		# Common non-actionable failure: provider account/payment status.
		if "Arrearage" in message or "overdue-payment" in message or "Access denied" in message:
			pytest.skip(f"Provider/account not available for '{model_name}': {message}")
		raise
	end_time = asyncio.get_event_loop().time()

	assert response == "Pong", f"Unexpected response: {response}"
	latency = round((end_time - start_time) * 1000, 2)
	print(f"✅ SUCCESS ({latency}ms)")
	print(f"   Model: {getattr(response, 'model', 'unknown')}")
	print(f"   Provider: {getattr(response, 'provider', 'unknown')}")
	print(f"   Response: {response}")
