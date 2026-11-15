import pytest
from pydantic import ValidationError

from siteguard.config.settings import RulesConfig, RuntimeConfig
from siteguard.inference.device import select_device


def test_config_rejects_invalid_and_unknown() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(inference_fps=0)
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"secret_typo": 1})
    with pytest.raises(ValidationError):
        RulesConfig(min_duration=10)


def test_explicit_cpu() -> None:
    assert select_device("cpu") == ("cpu", False)
    with pytest.raises(ValueError):
        select_device("cpu", True)
