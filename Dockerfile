# StockMate AI service image.
#
# The OCR models ship inside the rapidocr wheel, so the container needs no
# network at runtime and no API keys. Datasets are not baked in: they are large,
# they belong to their own authors, and the service does not need them. Only the
# benchmarks do, and those run outside the container.

FROM python:3.12-slim

# opencv needs these even in the headless build; without them the import of
# cv2 fails at runtime with a missing shared object rather than at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so that editing source does not invalidate the layer that
# takes minutes to build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stockmate/ ./stockmate/
COPY ui/ ./ui/
COPY data/ ./data/
COPY README.md LICENSE ./

# Run as a non-root user. The service reads bill images that arrive from
# outside, and an image decoder is exactly the sort of component that should not
# be running as root when it meets a malformed file.
RUN useradd --create-home --uid 10001 stockmate && chown -R stockmate:stockmate /app
USER stockmate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "stockmate.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
