import warnings
from pathlib import Path

import pytest

# Suppress known upstream deprecation warnings emitted by dependencies
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"dateutil\.tz",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"sqlalchemy\.sql\.sqltypes",
)


@pytest.fixture(autouse=True)
def _set_validator_config_path(monkeypatch):
    # Use local schema config during tests when package-level config is not set.
    config_dir = Path(__file__).resolve().parents[2] / "strata-fit-data-schema" / "config"
    if config_dir.exists():
        monkeypatch.setenv("CONFIG_PATH", str(config_dir))
