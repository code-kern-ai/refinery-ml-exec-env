ARG PARENT_IMAGE=registry.dev.kern.ai/code-kern-ai/refinery-parent-images:hardened-images-torch-cpu
ARG DHI_PYTHON_BUILD=dhi.io/python:3.11-debian12-dev

FROM ${PARENT_IMAGE} AS venv-source

FROM ${DHI_PYTHON_BUILD} AS builder

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

COPY --from=venv-source ${VENV_PATH} ${VENV_PATH}

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /inference && chown 65532:65532 /inference

COPY . .

FROM ${PARENT_IMAGE}

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

COPY --from=builder --chown=65532:65532 ${VENV_PATH} ${VENV_PATH}
COPY --from=builder --chown=65532:65532 /inference /inference
COPY --from=builder --chown=65532:65532 /app /app

USER nonroot

ENTRYPOINT ["/app/run.sh"]
