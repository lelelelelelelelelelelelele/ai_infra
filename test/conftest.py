import sys
from pathlib import Path

import pytest

from ai_infra.ai_infra import get_ai_models, load_secrets_from_env


def _ensure_repo_root_on_syspath() -> None:
	# repo_root == .../mcp
	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()

# Load secrets before collection to make model list/config resolution consistent.
load_secrets_from_env()


def pytest_addoption(parser):
	"""Register custom CLI options."""
	parser.addoption(
		"--models",
		action="store",
		default=None,
		help="Comma-separated list of models to test (e.g., gpt,gemini,deepseek)",
	)


def pytest_generate_tests(metafunc):
	"""Parametrize model-driven tests."""
	if "model_name" not in metafunc.fixturenames:
		return

	models_arg = metafunc.config.getoption("models")
	if models_arg:
		models_list = [m.strip() for m in models_arg.split(",") if m.strip()]
	else:
		models_list = get_ai_models()

	metafunc.parametrize("model_name", models_list)
