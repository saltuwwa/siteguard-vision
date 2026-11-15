import pytest

from siteguard.inference.runtime import require_ultralytics, ultralytics_version_status


@pytest.mark.parametrize("version,supported", [
    ("8.4.61", True), ("8.4.213", True), ("8.3.203", False), ("9.0.0", False),
    ("8.5.0rc1", False),
])
def test_version_range_is_not_execution_proof(monkeypatch: pytest.MonkeyPatch,
                                             version: str, supported: bool) -> None:
    monkeypatch.setattr("siteguard.inference.runtime.importlib.metadata.version", lambda name: version)
    status = ultralytics_version_status()
    assert status["version_supported"] is supported
    assert not status["execution_verified"]
    if supported:
        assert require_ultralytics() == version
    else:
        with pytest.raises(RuntimeError, match="Use ultralytics"):
            require_ultralytics()
