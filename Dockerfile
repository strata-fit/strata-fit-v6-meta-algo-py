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

# Core STRATA-FIT dependencies needed by the meta algorithm (pinned to commits)
RUN pip install \
    git+https://github.com/strata-fit/strata-fit-data-schema.git@53529a81518c43a57701a0c37c457af21018f8d5 \
    git+https://github.com/strata-fit/strata-fit-v6-imputation-py.git@0dc00d97aff9d6394d1accbc247a44e081168003 \
    git+https://github.com/strata-fit/strata-fit-v6-logistic-regression-py.git@b2d1b3597ea8cdf84763f86d88c382e3d352e14d \
    git+https://github.com/strata-fit/strata-fit-v6-km-py.git@4d1cc2fcc623a39da33d8a821a15c6a60de01397 


# Install this algorithm package
RUN pip install /app

# Runtime config
ENV PKG_NAME=${PKG_NAME}

# Use make for clarity
CMD ["make", "wrap-algorithm"]
