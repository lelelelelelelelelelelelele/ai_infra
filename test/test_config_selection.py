import os
import pathlib
import yaml

from ai_infra.ai_infra import _build_config_from_yaml


def test_main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    config_path = root / "ai_models.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    os.environ["DASHSCOPE_KEY"] = "dummy_dashscope_key"
    os.environ["IFLOW_KEY"] = "dummy_iflow_key"

    configs = _build_config_from_yaml("qwen-max", config_data)
    print(configs)
    assert configs, "No configs returned for qwen-max"
    assert configs[0]["model"] == "qwen-max"
    assert configs[1]["model"] == "qwen3-max"

    print("OK: providers list parsed with expected model names")


if __name__ == "__main__":
    test_main()
