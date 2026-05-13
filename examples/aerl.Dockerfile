FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENV AERL_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8765
CMD ["aerl"]
