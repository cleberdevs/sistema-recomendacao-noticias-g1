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

# Cria os scripts Python diretamente usando arquivos externos ou incorporados ao Dockerfile
# Todos os scripts de suporte são incluídos aqui
COPY <<-'EOT' /app/init_mlflow_db.py
import mlflow
import os
import sys
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# Remove arquivo de banco de dados existente, se houver
if os.path.exists("mlflow.db"):
    os.remove("mlflow.db")

# Cria uma nova store e inicializa
try:
    store = SqlAlchemyStore("sqlite:///mlflow.db")
    # Força a inicialização das tabelas
    store.get_experiment_by_name("Default")
    print("Banco de dados MLflow inicializado com sucesso")
except Exception as e:
    print(f"Erro ao inicializar banco de dados MLflow: {e}")
    sys.exit(1)
EOT

COPY <<-'EOT' /app/mlflow_manager.py
import os
import sys
import time
import signal
import subprocess
import requests

def start_mlflow_server():
    """Inicia o servidor MLflow e garante que continue rodando"""
    # Inicia servidor MLflow
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    # Abre arquivo de log
    log_file = open("mlflow.log", "w")
    
    # Inicia processo
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"MLflow iniciado com PID {proc.pid}")
    
    # Aguarda servidor estar pronto
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/")
            if response.status_code == 200:
                print(f"Servidor MLflow está rodando (tentativa {attempt+1}/{max_attempts})")
                return proc
        except requests.exceptions.ConnectionError:
            pass
        
        # Verifica se o processo ainda está rodando
        if proc.poll() is not None:
            print(f"Processo MLflow encerrou com código {proc.returncode}")
            log_file.close()
            with open("mlflow.log", "r") as f:
                print(f.read())
            sys.exit(1)
        
        time.sleep(1)
    
    print("Servidor MLflow falhou em iniciar no tempo esperado")
    proc.terminate()
    log_file.close()
    sys.exit(1)

def verify_model_registration():
    """Verifica se os modelos foram registrados no MLflow"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    print(f"Encontrados {len(experiments)} experimentos no MLflow")
    for exp in experiments:
        print(f"Experimento: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(exp.experiment_id)
        print(f"  Encontradas {len(runs)} execuções")
        
        for run in runs:
            print(f"  ID da Execução: {run.info.run_id}, Status: {run.info.status}")
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"  Artefatos: {len(artifacts)}")
            for artifact in artifacts:
                print(f"    {artifact.path}")
    
    return True

if __name__ == "__main__":
    # Verifica se está sendo chamado para verificar registro de modelo
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        success = verify_model_registration()
        sys.exit(0 if success else 1)
    
    # Inicia servidor MLflow
    mlflow_proc = start_mlflow_server()
    
    # Configura manipuladores de sinal para manter MLflow rodando
    def handle_signal(sig, frame):
        if sig in (signal.SIGINT, signal.SIGTERM):
            print("Sinal de terminação recebido, mas mantendo MLflow rodando para o pipeline")
        else:
            print(f"Sinal recebido {sig}")
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Mantém processo rodando até ser explicitamente encerrado
    try:
        while mlflow_proc.poll() is None:
            time.sleep(1)
        
        print(f"MLflow encerrou com código {mlflow_proc.returncode}")
        with open("mlflow.log", "r") as f:
            print(f.read())
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
        mlflow_proc.terminate()
EOT

COPY <<-'EOT' /app/test_pyspark.py
import os
import sys
import pyspark

# Exibe versões do Python e PySpark
print(f"Versão do Python: {sys.version}")
print(f"Versão do PySpark: {pyspark.__version__}")

# Define variáveis de ambiente
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get('PYSPARK_DRIVER_PYTHON')}")
print(f"sys.executable: {sys.executable}")

# Testa funcionalidade básica do PySpark
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("PySpark-Test") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable) \
        .getOrCreate()
    
    print("SparkSession criada com sucesso")
    
    # Cria dados de teste simples
    data = [(1, "teste")]
    df = spark.createDataFrame(data, ["id", "valor"])
    df.show()
    
    spark.stop()
    print("Teste do PySpark concluído com sucesso")
except Exception as e:
    print(f"Erro ao testar PySpark: {e}")
    sys.exit(1)
EOT

COPY <<-'EOT' /app/patch_mlflow.py
import os
import re
import sys

def patch_mlflow_integration():
    """Aplica patches em arquivos para melhorar a integração com MLflow"""
    files_to_patch = [
        "src/config/mlflow_config.py",
        "src/modelo/recomendador.py",
        "pipeline.py"
    ]
    
    for file_path in files_to_patch:
        if not os.path.exists(file_path):
            print(f"Aviso: {file_path} não encontrado")
            continue
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Garante que URI de rastreamento MLflow esteja configurada
        if "import mlflow" in content and "MLFLOW_TRACKING_URI" not in content:
            content = re.sub(
                r"import mlflow",
                "import mlflow\nimport os\nos.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'",
                content
            )
        
        # Melhora tratamento de erros MLflow
        if "mlflow.end_run" in content:
            content = re.sub(
                r"mlflow\.end_run\(.*\)",
                "try:\n        mlflow.end_run()\n    except Exception as e:\n        print(f\"Erro ao finalizar MLflow run: {e}\")",
                content
            )
        
        # Adiciona registro explícito de modelo
        if "mlflow.log_artifacts" in content and "mlflow.register_model" not in content:
            content = re.sub(
                r"mlflow\.log_artifacts\(.*\)",
                "\\g<0>\n        # Registra explicitamente o modelo\n        try:\n            mlflow.register_model(f\"runs:/{mlflow.active_run().info.run_id}/model\", \"sistema_recomendacao\")\n            print(\"Modelo registrado com sucesso no MLflow\")\n        except Exception as e:\n            print(f\"Erro ao registrar modelo no MLflow: {e}\")",
                content
            )
        
        with open(file_path, "w") as f:
            f.write(content)
        
        print(f"Arquivo {file_path} modificado com sucesso")

if __name__ == "__main__":
    patch_mlflow_integration()
EOT

# Aplica os patches do MLflow
RUN python patch_mlflow.py 

# Cria entrypoint.sh usando um método mais simples
COPY <<-'EOT' /app/entrypoint.sh
#!/bin/bash
echo "Verificando dependências..."
pip list | grep -E "mlflow|flask|pyspark|werkzeug|nltk"

# Verifica instalações do Python e Java
echo "Caminho do Python: $(which python)"
echo "Versão do Python: $(python --version)"
echo "Versão do Java:"
java -version
echo "JAVA_HOME=$JAVA_HOME"

# Testa configuração do PySpark
echo "Testando configuração do PySpark..."
python test_pyspark.py

# Garante versões corretas do Python para o PySpark
export PYSPARK_PYTHON=$(which python3.10)
export PYSPARK_DRIVER_PYTHON=$(which python3.10)
echo "PYSPARK_PYTHON=$PYSPARK_PYTHON"
echo "PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON"

# Inicializa banco de dados MLflow diretamente
echo "Inicializando banco de dados MLflow..."
python init_mlflow_db.py

# Inicia MLflow de forma a garantir que permaneça rodando durante a execução do pipeline
echo "Iniciando MLflow persistente..."
python mlflow_manager.py &
MLFLOW_PID=$!

# Aguarda o MLflow estar pronto
echo "Aguardando MLflow iniciar..."
max_attempts=30
attempt=0
mlflow_ready=false

while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt+1))
    echo "Verificando MLflow (tentativa $attempt/$max_attempts)..."
    
    if curl -s http://localhost:5000/ > /dev/null; then
        echo "MLflow iniciado com sucesso!"
        mlflow_ready=true
        break
    fi
    
    # Verifica se o processo MLflow ainda está rodando
    if ! kill -0 $MLFLOW_PID 2>/dev/null; then
        echo "ERRO: Processo do MLflow encerrou prematuramente."
        cat mlflow.log
        exit 1
    fi
    
    sleep 2
done

if [ "$mlflow_ready" = false ]; then
    echo "ERRO: MLflow não iniciou corretamente após várias tentativas."
    cat mlflow.log
    exit 1
fi

# Se solicitado, executa o pipeline primeiro
if [ "$RUN_PIPELINE" = "true" ]; then
    echo "Executando o pipeline..."
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    PYSPARK_PYTHON=$PYSPARK_PYTHON \
    PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON \
    python pipeline.py
    
    # Verifica status de saída do pipeline
    PIPELINE_STATUS=$?
    if [ $PIPELINE_STATUS -ne 0 ]; then
        echo "ERRO: Pipeline falhou com código de saída $PIPELINE_STATUS"
        exit $PIPELINE_STATUS
    fi
    
    # Verifica registro de modelos no MLflow
    echo "Verificando registro de modelos no MLflow..."
    python mlflow_manager.py verify
    
    # Verifica se os modelos foram criados
    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then
        echo "AVISO: Nenhum modelo foi encontrado após a execução do pipeline."
        echo "A API não será iniciada sem modelos treinados."
        exit 1
    fi
else
    # Verifica se os modelos existem
    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then
        echo "ERRO: Nenhum modelo encontrado em modelos/modelos_salvos/."
        echo "Execute o contêiner com -e RUN_PIPELINE=true para treinar os modelos primeiro"
        echo "ou monte um volume com modelos pré-treinados."
        exit 1
    fi
fi

# Função para lidar com o encerramento do contêiner de forma elegante
cleanup() {
    echo "Recebido sinal de encerramento. Encerrando processos..."
    if [ -n "$API_PID" ]; then
        echo "Encerrando API (PID: $API_PID)..."
        kill -TERM $API_PID 2>/dev/null || true
    fi
    
    if [ -n "$MLFLOW_PID" ]; then
        echo "Encerrando MLflow (PID: $MLFLOW_PID)..."
        kill -TERM $MLFLOW_PID 2>/dev/null || true
    fi
    
    echo "Processos encerrados, saindo..."
    exit 0
}

# Configura trap para cleanup ao parar o contêiner
trap cleanup SIGTERM SIGINT

# Arquivo para monitoramento de logs da API
touch api.log

# Inicia o servidor da API em segundo plano
echo "Iniciando API..."
./scripts/start_api.sh > api.log 2>&1 &
API_PID=$!

# Aguarda a API iniciar
echo "Aguardando API iniciar..."
sleep 5

# Verifica se a API está rodando
if ! kill -0 $API_PID 2>/dev/null; then
    echo "ERRO: API falhou ao iniciar. Verificando logs:"
    cat api.log
    exit 1
fi

echo "==============================================="
echo "MLflow UI está rodando em: http://localhost:5000"
echo "API está rodando em: http://localhost:8000"
echo "==============================================="

# Monitora ambos os processos e mantém o contêiner rodando
echo "Monitorando processos..."
while true; do
    # Verifica se o MLflow ainda está rodando
    if ! kill -0 $MLFLOW_PID 2>/dev/null; then
        echo "ALERTA: MLflow encerrou inesperadamente. Tentando reiniciar..."
        python mlflow_manager.py &
        MLFLOW_PID=$!
        echo "MLflow reiniciado com PID: $MLFLOW_PID"
    fi
    
    # Verifica se a API ainda está rodando
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "ALERTA: API encerrou inesperadamente. Tentando reiniciar..."
        ./scripts/start_api.sh > api.log 2>&1 &
        API_PID=$!
        echo "API reiniciada com PID: $API_PID"
    fi
    
    # Se ambos os processos falharem, encerra o contêiner
    if ! kill -0 $MLFLOW_PID 2>/dev/null && ! kill -0 $API_PID 2>/dev/null; then
        echo "ERRO: Tanto o MLflow quanto a API falharam múltiplas vezes. Encerrando contêiner."
        echo "Verificando logs do MLflow:"
        cat mlflow.log
        echo "Verificando logs da API:"
        cat api.log
        exit 1
    fi
    
    # Dorme e verifica novamente
    sleep 10
done
EOT

# Torna o entrypoint executável
RUN chmod +x /app/entrypoint.sh

# Define o entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]