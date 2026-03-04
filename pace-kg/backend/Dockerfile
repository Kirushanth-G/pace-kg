FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for marker-pdf and spacy
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
