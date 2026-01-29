import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `algorithms.*` imports work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynaconf import Dynaconf
from vantage6.client import Client
from algorithms.config import config
from algorithms.types import Vantage6AlgorithmConfig

# Load algorithm presets from settings/algorithms.toml
algorithm_settings = Dynaconf(settings_files=["settings/algorithms.toml"])


def run_task(algorithm_config: Vantage6AlgorithmConfig):
    """Create a vantage6 task using the provided algorithm configuration."""
    client = Client(config.server_url, config.server_port, config.server_api)
    client.authenticate(username=config.username, password=config.password)

    if getattr(config, "organization_key", None):
        client.setup_encryption(config.organization_key)

    organizations_data = client.organization.list().get("data", [])
    if not organizations_data:
        raise RuntimeError("No organizations found!")

    collaborations = client.collaboration.list().get("data", [])
    if not collaborations:
        raise RuntimeError("No collaborations found!")

    task = client.task.create(
        image=algorithm_config.image,
        name=algorithm_config.name,
        description=algorithm_config.description,
        input_=algorithm_config.input_.model_dump(by_alias=True),
        organizations=algorithm_config.organizations,
        collaboration=algorithm_config.collaboration,
        databases=algorithm_config.databases,
    )

    task_id = task.get("id")
    client.wait_for_results(task_id)
    results = client.result.get(task_id)
    return results


def load_algorithm_config(name: str) -> Vantage6AlgorithmConfig:
    """Load an algorithm configuration by name from algorithms.toml."""
    raw_config = algorithm_settings.get(name)
    if not raw_config:
        raise ValueError(f"Algorithm '{name}' not found in settings/algorithms.toml")
    return Vantage6AlgorithmConfig(**raw_config)


if __name__ == "__main__":
    # Choose which algorithm to run via ALGORITHM env var.
    algo_name = os.environ.get("ALGORITHM", "simi_gaussian")
    algo_config = load_algorithm_config(algo_name)

    results = run_task(algo_config)
    print(json.dumps({f"{algo_name}_results": results}, indent=2))
