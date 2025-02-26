# Stage 1: Base Python
FROM python:3.9-slim as python-base

# Configuração de ambiente não interativo
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 2: Builder - para compilar dependências
FROM python-base as builder

# Instalar dependências do sistema necessárias para compilação
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        default-jdk \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /build

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 3: Runtime final
FROM python-base as runtime

# Labels informativos
LABEL maintainer="Sistema de Recomendação" \
      version="1.0" \
      description="Sistema de recomendação de notícias"

# Instalar dependências do sistema necessárias em runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
        libgomp1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -m -r appuser

# Criar diretórios necessários
RUN mkdir -p /app/dados/{brutos,processados} \
    /app/modelos/modelos_salvos \
    /app/logs \
    /app/mlflow-artifacts \
    /app/spark-logs \
    /app/checkpoints \
    && chown -R appuser:appuser /app

# Copiar wheels do builder e instalar dependências
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/*

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app \
    SPARK_LOCAL_DIRS=/app/spark-logs \
    SPARK_WORKER_DIR=/app/spark-logs \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    FLASK_APP=src/api/app.py \
    FLASK_ENV=production \
    MODEL_VERSION=v1.0

# Mudar para o usuário não-root
USER appuser
WORKDIR /app

# Copiar código fonte
COPY --chown=appuser:appuser . .

# Expor portas
EXPOSE 8000 5000

# Script de inicialização
COPY --chown=appuser:appuser scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/sistema/saude || exit 1

# Comando de entrada
ENTRYPOINT ["/app/docker-entrypoint.sh"]