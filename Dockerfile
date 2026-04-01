# Basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:4.13

# Algorithm package name (as in setup.py)
ARG PKG_NAME="strata_fit_v6_meta_algo_py"
# List of extra git-based packages, space-separated
# Example at build:
#   docker build \
#     --build-arg EXTRA_PIP_PACKAGES="git+https://github.com/org/alg-logreg.git@a1b2c3d4 git+https://github.com/org/alg-km.git@11223344" \
#     -t v6-logreg .
ARG EXTRA_PIP_PACKAGES=""
ENV EXTRA_PIP_PACKAGES=${EXTRA_PIP_PACKAGES}

# Make sure git + make are available (git needed for git+https pip urls)
RUN apt-get update && apt-get install -y git make && rm -rf /var/lib/apt/lists/*

# Copy everything in
WORKDIR /app
COPY . /app

# Install extra algorithm packages via Makefile
RUN make install-algo-packages

# Shared dependencies (install core once to avoid resolver conflicts across git refs)
RUN pip install \
    "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz" \
    dynaconf \
    fastapi \
    gunicorn \
    uvicorn \
    python-multipart \
    polars \
    pyarrow \
    numpy \
    pandas \
    scipy \
    scikit-learn

# Core STRATA-FIT algorithm packages (pinned). Use --no-deps to keep core pin consistent.
RUN pip install \
    --no-deps \
    "strata_fit_v6_data_validator_py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz" \
    "strata-fit-v6-imputation-py @ https://github.com/strata-fit/strata-fit-v6-imputation-py/archive/0db3b1d023505b5f196320ff6d2fcb5ce29aafa8.tar.gz" \
    "coxph @ https://github.com/strata-fit/v6-coxph-py/archive/e0e8a98b68207c0e3756628a975dc8a2cfa95f5f.tar.gz" \
    "v6_sklearn_linear_py @ https://github.com/strata-fit/strata-fit-v6-sklearn-linear-py/archive/c9d9fc2faf0f4f27224cf786b7979a9f2a4662ce.tar.gz" \
    "strata_fit_v6_km_py @ https://github.com/strata-fit/strata-fit-v6-km-py/archive/4d1cc2fcc623a39da33d8a821a15c6a60de01397.tar.gz"

# Install this algorithm package without re-resolving already pinned dependencies.
RUN pip install --no-deps /app

# Runtime config
ENV PKG_NAME=${PKG_NAME}

# Use make for clarity
CMD ["make", "wrap-algorithm"]
