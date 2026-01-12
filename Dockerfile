# 1. Utiliser une image Python légère (Conforme Section 3.1 & 3.3) [cite: 12, 23]
FROM python:3.10-slim

# 2. Définir des variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Installer les dépendances système indispensables pour ctransformers et la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Installation des dépendances (Optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copier le reste du code [cite: 24]
# Note : Les données gérées par DVC seront copiées ici si elles sont présentes localement
COPY . .

# 6. Créer un utilisateur non-root pour la sécurité (Bonne pratique MLOps)
RUN useradd -m mluser && chown -R mluser /app
USER mluser

# 7. Exposer le port Streamlit [cite: 44]
EXPOSE 8501

# 8. Commande de lancement (ENTRYPOINT pour plus de robustesse sur Azure) [cite: 44]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]