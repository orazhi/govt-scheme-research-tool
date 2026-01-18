FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER root

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    libxml2-dev \
    libxslt-dev \
    poppler-utils \
    tesseract-ocr \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
