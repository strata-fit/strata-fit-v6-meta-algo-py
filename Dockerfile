# Basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:4.13


# Make sure git + make are available (git needed for git+https pip urls)
RUN apt-get update && apt-get install -y git make && rm -rf /var/lib/apt/lists/*

# Copy everything in 
WORKDIR /app
COPY . /app


# Install this algorithm package
RUN pip install .

# Runtime config
ENV PKG_NAME="strata_fit_v6_meta_algo_py"

# Use make for clarity
CMD ["make", "wrap-algorithm"]
