# Usar imagem base do Python
FROM python:3.10-slim

# Argumentos de build
ARG DEBIAN_FRONTEND=noninteractive
ARG JAVA_VERSION=17

# Configurar ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64 \
    PYTHONPATH=/app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Adicionar repositório para versões mais recentes do Java
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    && \
    # Instalar OpenJDK 17
    apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Verificar instalação do Java
RUN java -version

# Criar diretórios necessários
WORKDIR /app

# Atualizar pip e instalar dependências básicas
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir \
    setuptools==68.2.2 \
    wheel==0.41.2

# Instalar TensorFlow e dependências primárias primeiro
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    tensorflow-io==0.32.0 \
    tensorflow-hub==0.14.0 \
    protobuf==3.20.3

# Instalar dependências principais
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    mlflow==2.3.0 \
    Flask==2.2.5 \
    Werkzeug==2.2.3 \
    flask-restx==1.1.0 \
    python-dotenv==0.19.0 \
    swagger-ui-bundle==0.0.9 \
    pyspark==3.2.0 \
    pyarrow==7.0.0 \
    nltk==3.8.1

# Instalar dependências NLTK
RUN python3 -c "import nltk; nltk.download('stopwords')"

# Copiar código fonte
COPY . .

# Criar diretórios e configurar permissões
RUN mkdir -p \
    dados/brutos \
    dados/processados \
    logs \
    mlflow-artifacts \
    modelos/modelos_salvos \
    checkpoints \
    spark-logs && \
    chmod -R 777 /app

# Verificar instalações
RUN echo "print('Verificando dependências...')" > verify_deps.py && \
    echo "import tensorflow as tf" >> verify_deps.py && \
    echo "import mlflow" >> verify_deps.py && \
    echo "import numpy as np" >> verify_deps.py && \
    echo "import pandas as pd" >> verify_deps.py && \
    echo "import sklearn" >> verify_deps.py && \
    echo "import pyspark" >> verify_deps.py && \
    echo "print('TensorFlow version:', tf.__version__)" >> verify_deps.py && \
    echo "print('MLflow version:', mlflow.__version__)" >> verify_deps.py && \
    echo "print('NumPy version:', np.__version__)" >> verify_deps.py && \
    echo "print('Pandas version:', pd.__version__)" >> verify_deps.py && \
    echo "print('Scikit-learn version:', sklearn.__version__)" >> verify_deps.py && \
    echo "print('PySpark version:', pyspark.__version__)" >> verify_deps.py && \
    python3 verify_deps.py && \
    rm verify_deps.py

# Copiar e configurar spark-defaults.conf
COPY src/config/spark-defaults.conf /opt/spark/conf/

# Configurar variáveis de ambiente
ENV FLASK_APP=src/api/app.py \
    FLASK_ENV=production \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MLFLOW_EXPERIMENT_NAME=recomendador_noticias \
    SPARK_LOCAL_DIRS=/tmp \
    SPARK_WORKER_DIR=/tmp \
    LOG_LEVEL=INFO \
    MODEL_VERSION=v1.0

# Expor portas
EXPOSE 8000 5000

# Script de entrada
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/sistema/saude || exit 1

# Entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]
