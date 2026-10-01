# PAPER-TRADING image. Contains no exchange credentials and no code able to place real orders.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY cryptoalgo ./cryptoalgo
RUN pip install --no-cache-dir -e ".[live-data]"
COPY config ./config
COPY scripts ./scripts
COPY reports ./reports
VOLUME ["/app/runtime"]
EXPOSE 8080
HEALTHCHECK --interval=2m --timeout=10s CMD python scripts/healthcheck.py --max-age 600 || exit 1
CMD ["python", "scripts/run_paper.py", "--dashboard"]
