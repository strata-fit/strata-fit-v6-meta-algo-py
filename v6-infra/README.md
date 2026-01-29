# Vantage6 Demo Network Infrastructure

This repository provides a demo setup for running a [vantage6](https://vantage6.ai/) server, nodes, and UI locally for testing and development.

## Getting Started

1. **Clone** this repository.
2. **Check** or **edit** `config.env` to set any desired defaults (e.g., vantage6 version, Docker registry, UI port, etc.).
3. **Run** the setup script:
    ```bash
    # Development usage (won't stop on errors, opens browser)
    ENVIRONMENT=DEV ./setup.sh
    ```
   or
    ```bash
    # CI usage (stops on errors, no browser launch)
    ENVIRONMENT=CI ./setup.sh
    ```
    If you omit `ENVIRONMENT=...`, it falls back to whatever is in `config.env`.

4. **Verify** containers are running:
    ```bash
    docker ps
    ```
    You should see the vantage6 server, nodes, and UI container.

5. **Interact** with vantage6 (e.g., run an algorithm). The vantage6 UI can be accessed at http://localhost (configurable in `config.env`).

6. **Stop** and **remove** all containers:
    ```bash
    # Use the same ENVIRONMENT mode you started with, if desired
    ENVIRONMENT=DEV ./shutdown.sh
    ```
    This tears down the vantage6 environment and cleans up leftover files.

## Meta Algorithm Demo (validation → imputation → LR)
- Data for the meta algorithm lives in `infrastructure/data/meta` and follows the `PatientData` schema from `strata-fit-data-schema`. Missing values are present in `DAS28`, `CRP`, and `HAQ` for the gamma site to exercise imputation.
- Build the meta image from the repo root (push/tag as needed for your registry). The Dockerfile already installs pinned GitHub commits of `strata_fit_v6_data_validator_py`, `strata_fit_v6_imputation_py`, and `v6_logistic_regression_py`:
  - `docker build -t strata-fit-v6-meta-algo:latest -f Dockerfile .`
- A preset is available in `algorithms/settings/algorithms.toml` under `[meta_generic]`.
- Run the preset once the demo infrastructure is up:
  - `ALGORITHM=meta_generic python v6-infra/algorithms/run.py`
- Quick local smoke test without Docker using the vantage6 `MockAlgorithmClient`:
  - `pytest tests/test_meta_mock_client.py`
