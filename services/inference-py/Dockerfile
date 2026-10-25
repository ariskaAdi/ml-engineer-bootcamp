# Python inference service. Build dari folder services/:
#   docker build -t bank-inference ./inference-py
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# 1) dependency dulu: layer ini di-cache selama requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) kode + model package (hasil scripts/export_model.py)
COPY app ./app
COPY model ./model

# Jangan jalan sebagai root di dalam container
RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

ENV MODEL_DIR=/srv/model \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/ready')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
