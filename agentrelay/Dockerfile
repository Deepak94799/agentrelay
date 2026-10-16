FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir ".[server]" && useradd -m app && chown -R app /app
USER app
ENV AGENTRELAY_HOME=/app/examples
EXPOSE 8000
CMD ["agentrelay", "serve", "--host", "0.0.0.0", "--port", "8000"]
