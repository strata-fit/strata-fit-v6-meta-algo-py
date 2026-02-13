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
    git+https://github.com/strata-fit/strata-fit-data-schema.git@63ecfde03d8a0b80d1d39337e89f3f8b90b7b1cb \
    #git+https://github.com/strata-fit/strata-fit-v6-imputation-py.git@848bb4bf1ef8b905d0b437b95533762f9b0bb9482299f4adcb3cb3903a4686f7 \
    git+https://github.com/strata-fit/strata-fit-v6-logistic-regression-py.git@53c3d0ab418fc09fed11ac79c3a19abe3ef01d70 \
    git+https://github.com/strata-fit/strata-fit-v6-km-py.git@4d1cc2fcc623a39da33d8a821a15c6a60de01397 

# Install this algorithm package
RUN pip install /app

# Runtime config
ENV PKG_NAME=${PKG_NAME}

# Use make for clarity
CMD ["make", "wrap-algorithm"]
