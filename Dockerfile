FROM harbor2.vantage6.ai/infrastructure/algorithm-base:4.13

WORKDIR /app
COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install the runtime dependencies the meta-algorithm imports directly.
RUN pip install --no-cache-dir \
    dynaconf \
    numpy \
    pandas \
    pydantic \
    PyJWT \
    requests \
    scikit-learn \
    scipy

# Install pinned algorithm dependencies without
# re-resolving/transitively drifting already pinned runtime dependencies.
RUN pip install --no-cache-dir --no-deps \
    "strata-fit-v6-data-validator-py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz"
RUN pip install --no-cache-dir --no-deps \
    "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz"

# Install this algorithm package without re-resolving transitive dependencies.
RUN pip install --no-cache-dir --no-deps .

# Runtime config
ENV PKG_NAME="strata_fit_v6_meta_algo_py"

CMD ["python", "-m", "strata_fit_v6_meta_algo_py.container"]
