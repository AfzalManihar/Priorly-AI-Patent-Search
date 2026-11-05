FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY backend/requirements.txt backend/requirements-ml.txt ./
ARG WITH_ML=false
RUN pip install -r $([ "$WITH_ML" = "true" ] && echo requirements-ml.txt || echo requirements.txt)

COPY backend /srv/backend
COPY frontend /srv/frontend

# Run as an unprivileged user; /srv/backend/data is the only writable path (SQLite, sample data).
RUN useradd --system --uid 10001 --home /srv priorly \
    && mkdir -p /srv/backend/data && chown -R priorly /srv/backend/data
USER priorly

WORKDIR /srv/backend
ENV FRONTEND_DIR=/srv/frontend DB_PATH=/srv/backend/data/priorly.db LOG_LEVEL=INFO
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
