# Imagem base: Ubuntu Focal (20.04)
FROM ubuntu:focal

# Evita prompts durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema, Python 3.10 e OpenJDK 21
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    net-tools \
    procps \
    sqlite3 \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:openjdk-r/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    openjdk-21-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala pip para Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Cria symlinks para conveniência
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# Define o diretório de trabalho
WORKDIR /app

# Copia primeiro o projeto inteiro
COPY . .

# Torna os scripts executáveis
RUN chmod +x scripts/*.sh

# Executa o script setup_environment.sh para configurar o ambiente corretamente
RUN ./scripts/setup_environment.sh

# Cria diretórios necessários (se ainda não existirem do setup_environment.sh)
RUN mkdir -p dados/brutos/itens
RUN mkdir -p modelos/modelos_salvos
RUN mkdir -p mlruns artifacts

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    FLASK_APP=src/api/app.py \
    FLASK_DEBUG=0 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PYTHONPATH=/app \
    PYSPARK_PYTHON=python3.10 \
    PYSPARK_DRIVER_PYTHON=python3.10

# Expõe portas
# 8000: API Flask
# 5000: Servidor MLflow
EXPOSE 8000 5000

# Abordagem alternativa: Copiar os scripts de um diretório temporário
COPY docker-scripts/ /tmp/scripts/

# Move os scripts para seus locais corretos
RUN cp /tmp/scripts/init_mlflow_db.py /app/ && \
    cp /tmp/scripts/mlflow_manager.py /app/ && \
    cp /tmp/scripts/test_pyspark.py /app/ && \
    cp /tmp/scripts/mlflow_wrapper.py /app/ && \
    cp /tmp/scripts/entrypoint.sh /app/ && \
    chmod +x /app/mlflow_wrapper.py /app/entrypoint.sh

# Define o entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]