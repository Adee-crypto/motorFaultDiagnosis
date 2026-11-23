# One image for the server and the nodes (step 7).
# Built by docker-compose.yml from the repo root, so it can include the
# diagnosis code (predict_motor.py, ml/, ...) next to monitoring/.
# The recordings (Loaded/, Unloaded/) are NOT copied in: compose mounts them
# read-only when the nodes start, so the image stays small.
ARG BASE=python:3.12-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libraries first: this layer is reused as long as the requirement lists don't change
COPY monitoring/requirements.txt monitoring/requirements-diagnosis.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-diagnosis.txt -r /tmp/requirements.txt

# then the code (only what Dockerfile.dockerignore lets through)
COPY . /app/

# the server's dashboard and API
EXPOSE 8000

# no default program: compose says what each container runs
CMD ["python", "monitoring/server/server.py", "--broker", "broker"]
