FROM python:3.11-slim
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY . .
CMD ["python", "scripts/demo_retrieval.py"]
