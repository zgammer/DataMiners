FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# веса кладём в ./weights и ./artifacts заранее (см. README)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
