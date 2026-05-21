# ──────────────────────────────────────────────────────────────
# Dockerfile — Analyseur Académique IA (Backend)
# Optimisé pour le déploiement sur Render.com (gratuit)
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Empêche Python d'écrire des fichiers .pyc et active la sortie non-bufferisée
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système requises par certains paquets Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python en premier (optimisation du cache de couches)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p uploads chroma_data logs app/data

# Exposer le port (Render injecte la variable d'env PORT)
EXPOSE 8000

# Démarrer le serveur — Render fournit $PORT, par défaut 8000 pour les tests locaux
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
