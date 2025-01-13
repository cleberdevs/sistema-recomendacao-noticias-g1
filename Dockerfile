# Dockerfile

# Imagem base
FROM python:3.8-slim

# Argumentos de build
ARG ENVIRONMENT=producao

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=src/api/app.py \
    FLASK_ENV=${ENVIRONMENT} \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    PORT=8000

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p mlflow-artifacts logs \
    && chmod -R 777 mlflow-artifacts logs

# Script de health check
COPY scripts/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Expor portas
EXPOSE 8000 5000

# Copiar e tornar executável o script de inicialização
COPY start_services.sh .
RUN chmod +x start_services.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /healthcheck.sh

# Comando para iniciar serviços
CMD ["./start_services.sh"]