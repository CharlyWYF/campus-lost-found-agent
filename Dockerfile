FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    LF_WEB_HOST=0.0.0.0 \
    LF_WEB_PORT=8000

# python:3.12-slim 已自带 ca-certificates / netbase / tzdata，
# 不再 apt-get，避免 NAS 上构建时卡在 Debian 源。
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY server.py ./
COPY agent ./agent
COPY web ./web

RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "server.py"]
