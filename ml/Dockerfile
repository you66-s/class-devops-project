FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système nécessaires (wordcloud, matplotlib)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le script d'entraînement
COPY train.py .

# Lancer le modèle
CMD ["python", "train.py"]
