# The agent's shell. It only sees /workspace and reaches the warehouse over the network.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir snowflake-connector-python pandas \
    && useradd -m agent && mkdir /workspace && chown agent /workspace
USER agent
WORKDIR /workspace
