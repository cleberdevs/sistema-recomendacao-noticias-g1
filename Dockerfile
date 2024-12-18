FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Criar diretórios necessários
RUN mkdir -p mlflow-artifacts logs

# Expor portas para Flask e MLflow
EXPOSE 8000
EXPOSE 5000

# Variáveis de ambiente
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=producao
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Tornar script de inicialização executável
RUN chmod +x start_services.sh

# Iniciar serviços
CMD ["./start_services.sh"]
