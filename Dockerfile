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
    "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/refs/heads/main.tar.gz" \
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
    git+https://github.com/strata-fit/strata-fit-data-schema.git@8812c7552a08b0411e6e0233d336919d1e2460d9 \
    git+https://github.com/strata-fit/strata-fit-v6-imputation-py.git@07e4599a4e0f9539a18e8de2285dcc979507e8d1 \
    git+https://github.com/MaastrichtU-CDS/v6-coxph.git@e0e8a98b68207c0e3756628a975dc8a2cfa95f5f

# Install this algorithm package
RUN pip install /app

# Runtime config
ENV PKG_NAME=${PKG_NAME}

# Use make for clarity
CMD ["make", "wrap-algorithm"]
