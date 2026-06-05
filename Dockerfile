FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Shared dependencies and core pinned to immutable references
RUN pip install \
    "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz" \
    dynaconf \
    fastapi \
    gunicorn \
    uvicorn \
    python-multipart \
    pyarrow \
    requests \
    PyJWT \
    pydantic

# Install git-sourced algorithm dependencies from immutable commits without
# re-resolving/transitively drifting already pinned runtime dependencies.
RUN pip install --no-deps \
    "strata-fit-v6-data-validator-py @ git+https://github.com/strata-fit/strata-fit-data-schema.git@c77d319b6539bdc48314738981b5bde478d2bacd"

# Install this algorithm package without re-resolving transitive dependencies.
RUN pip install --no-deps .

# Runtime config
ENV PKG_NAME="strata_fit_v6_meta_algo_py"

CMD ["python", "-m", "strata_fit_v6_meta_algo_py.container"]
