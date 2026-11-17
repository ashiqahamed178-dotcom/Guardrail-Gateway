FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway gateway
COPY scripts scripts
COPY evals evals
# Train the TF-IDF detector at build time (downloads the public deepset dataset, ~50 KB)
RUN python scripts/train_tfidf.py
ENV GUARD_DB=/tmp/guard.db PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT}"]
