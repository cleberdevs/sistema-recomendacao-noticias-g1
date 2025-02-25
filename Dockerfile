# Começando com uma imagem que já contém Java
FROM openjdk:11-slim

# Instalar Python e utilitários
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    procps \
    curl \
    netcat-openbsd \
    net-tools \
    unzip \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Definir variáveis de ambiente do Spark
ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.5-src.zip
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=/usr/bin/python3

# Instalar o Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Diretório de trabalho
WORKDIR /app

# Copiar todo o diretório do projeto
COPY . /app/

# Instalar dependências Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

# Garantir que os scripts shell sejam executáveis
RUN chmod +x /app/*.sh /app/scripts/*.sh

# Expor portas necessárias (MLflow e API)
EXPOSE 5000 8000

# Definir o script de entrada
ENTRYPOINT ["/app/entrypoint.sh"]