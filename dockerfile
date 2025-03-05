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

# Cria todos os arquivos necessários usando 'echo' diretamente para evitar problemas de indentação
RUN echo 'import mlflow' > /app/init_mlflow_db.py && \
    echo 'import os' >> /app/init_mlflow_db.py && \
    echo 'import sys' >> /app/init_mlflow_db.py && \
    echo 'from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore' >> /app/init_mlflow_db.py && \
    echo '' >> /app/init_mlflow_db.py && \
    echo '# Remove arquivo de banco de dados existente, se houver' >> /app/init_mlflow_db.py && \
    echo 'if os.path.exists("mlflow.db"):' >> /app/init_mlflow_db.py && \
    echo '    os.remove("mlflow.db")' >> /app/init_mlflow_db.py && \
    echo '' >> /app/init_mlflow_db.py && \
    echo '# Cria uma nova store e inicializa' >> /app/init_mlflow_db.py && \
    echo 'try:' >> /app/init_mlflow_db.py && \
    echo '    store = SqlAlchemyStore("sqlite:///mlflow.db")' >> /app/init_mlflow_db.py && \
    echo '    # Força a inicialização das tabelas' >> /app/init_mlflow_db.py && \
    echo '    store.get_experiment_by_name("Default")' >> /app/init_mlflow_db.py && \
    echo '    print("Banco de dados MLflow inicializado com sucesso")' >> /app/init_mlflow_db.py && \
    echo 'except Exception as e:' >> /app/init_mlflow_db.py && \
    echo '    print(f"Erro ao inicializar banco de dados MLflow: {e}")' >> /app/init_mlflow_db.py && \
    echo '    sys.exit(1)' >> /app/init_mlflow_db.py

RUN echo 'import os' > /app/test_pyspark.py && \
    echo 'import sys' >> /app/test_pyspark.py && \
    echo 'import pyspark' >> /app/test_pyspark.py && \
    echo '' >> /app/test_pyspark.py && \
    echo '# Exibe versões do Python e PySpark' >> /app/test_pyspark.py && \
    echo 'print(f"Versão do Python: {sys.version}")' >> /app/test_pyspark.py && \
    echo 'print(f"Versão do PySpark: {pyspark.__version__}")' >> /app/test_pyspark.py && \
    echo '' >> /app/test_pyspark.py && \
    echo '# Define variáveis de ambiente' >> /app/test_pyspark.py && \
    echo 'os.environ["PYSPARK_PYTHON"] = sys.executable' >> /app/test_pyspark.py && \
    echo 'os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable' >> /app/test_pyspark.py && \
    echo '' >> /app/test_pyspark.py && \
    echo 'print(f"PYSPARK_PYTHON: {os.environ.get(\"PYSPARK_PYTHON\")}")' >> /app/test_pyspark.py && \
    echo 'print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get(\"PYSPARK_DRIVER_PYTHON\")}")' >> /app/test_pyspark.py && \
    echo 'print(f"sys.executable: {sys.executable}")' >> /app/test_pyspark.py && \
    echo '' >> /app/test_pyspark.py && \
    echo '# Testa funcionalidade básica do PySpark' >> /app/test_pyspark.py && \
    echo 'try:' >> /app/test_pyspark.py && \
    echo '    from pyspark.sql import SparkSession' >> /app/test_pyspark.py && \
    echo '    spark = SparkSession.builder.appName("PySpark-Test").getOrCreate()' >> /app/test_pyspark.py && \
    echo '    print("SparkSession criada com sucesso")' >> /app/test_pyspark.py && \
    echo '    data = [(1, "teste")]' >> /app/test_pyspark.py && \
    echo '    df = spark.createDataFrame(data, ["id", "valor"])' >> /app/test_pyspark.py && \
    echo '    df.show()' >> /app/test_pyspark.py && \
    echo '    spark.stop()' >> /app/test_pyspark.py && \
    echo '    print("Teste do PySpark concluído com sucesso")' >> /app/test_pyspark.py && \
    echo 'except Exception as e:' >> /app/test_pyspark.py && \
    echo '    print(f"Erro ao testar PySpark: {e}")' >> /app/test_pyspark.py && \
    echo '    sys.exit(1)' >> /app/test_pyspark.py

RUN echo 'import os' > /app/mlflow_manager.py && \
    echo 'import sys' >> /app/mlflow_manager.py && \
    echo 'import time' >> /app/mlflow_manager.py && \
    echo 'import signal' >> /app/mlflow_manager.py && \
    echo 'import subprocess' >> /app/mlflow_manager.py && \
    echo 'import requests' >> /app/mlflow_manager.py && \
    echo '' >> /app/mlflow_manager.py && \
    echo 'def start_mlflow_server():' >> /app/mlflow_manager.py && \
    echo '    """Inicia o servidor MLflow e garante que continue rodando"""' >> /app/mlflow_manager.py && \
    echo '    cmd = ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./artifacts", "--host", "0.0.0.0", "--port", "5000"]' >> /app/mlflow_manager.py && \
    echo '    log_file = open("mlflow.log", "w")' >> /app/mlflow_manager.py && \
    echo '    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)' >> /app/mlflow_manager.py && \
    echo '    print(f"MLflow iniciado com PID {proc.pid}")' >> /app/mlflow_manager.py && \
    echo '    max_attempts = 30' >> /app/mlflow_manager.py && \
    echo '    for attempt in range(max_attempts):' >> /app/mlflow_manager.py && \
    echo '        try:' >> /app/mlflow_manager.py && \
    echo '            response = requests.get("http://localhost:5000/")' >> /app/mlflow_manager.py && \
    echo '            if response.status_code == 200:' >> /app/mlflow_manager.py && \
    echo '                print(f"Servidor MLflow está rodando (tentativa {attempt+1}/{max_attempts})")' >> /app/mlflow_manager.py && \
    echo '                return proc' >> /app/mlflow_manager.py && \
    echo '        except requests.exceptions.ConnectionError:' >> /app/mlflow_manager.py && \
    echo '            pass' >> /app/mlflow_manager.py && \
    echo '        if proc.poll() is not None:' >> /app/mlflow_manager.py && \
    echo '            print(f"Processo MLflow encerrou com código {proc.returncode}")' >> /app/mlflow_manager.py && \
    echo '            log_file.close()' >> /app/mlflow_manager.py && \
    echo '            with open("mlflow.log", "r") as f:' >> /app/mlflow_manager.py && \
    echo '                print(f.read())' >> /app/mlflow_manager.py && \
    echo '            sys.exit(1)' >> /app/mlflow_manager.py && \
    echo '        time.sleep(1)' >> /app/mlflow_manager.py && \
    echo '    print("Servidor MLflow falhou em iniciar no tempo esperado")' >> /app/mlflow_manager.py && \
    echo '    proc.terminate()' >> /app/mlflow_manager.py && \
    echo '    log_file.close()' >> /app/mlflow_manager.py && \
    echo '    sys.exit(1)' >> /app/mlflow_manager.py && \
    echo '' >> /app/mlflow_manager.py && \
    echo 'def verify_model_registration():' >> /app/mlflow_manager.py && \
    echo '    """Verifica se os modelos foram registrados no MLflow"""' >> /app/mlflow_manager.py && \
    echo '    import mlflow' >> /app/mlflow_manager.py && \
    echo '    from mlflow.tracking import MlflowClient' >> /app/mlflow_manager.py && \
    echo '    client = MlflowClient()' >> /app/mlflow_manager.py && \
    echo '    experiments = client.search_experiments()' >> /app/mlflow_manager.py && \
    echo '    print(f"Encontrados {len(experiments)} experimentos no MLflow")' >> /app/mlflow_manager.py && \
    echo '    for exp in experiments:' >> /app/mlflow_manager.py && \
    echo '        print(f"Experimento: {exp.name} (ID: {exp.experiment_id})")' >> /app/mlflow_manager.py && \
    echo '        runs = client.search_runs(exp.experiment_id)' >> /app/mlflow_manager.py && \
    echo '        print(f"  Encontradas {len(runs)} execuções")' >> /app/mlflow_manager.py && \
    echo '        for run in runs:' >> /app/mlflow_manager.py && \
    echo '            print(f"  ID da Execução: {run.info.run_id}, Status: {run.info.status}")' >> /app/mlflow_manager.py && \
    echo '            artifacts = client.list_artifacts(run.info.run_id)' >> /app/mlflow_manager.py && \
    echo '            print(f"  Artefatos: {len(artifacts)}")' >> /app/mlflow_manager.py && \
    echo '            for artifact in artifacts:' >> /app/mlflow_manager.py && \
    echo '                print(f"    {artifact.path}")' >> /app/mlflow_manager.py && \
    echo '    return True' >> /app/mlflow_manager.py && \
    echo '' >> /app/mlflow_manager.py && \
    echo 'if __name__ == "__main__":' >> /app/mlflow_manager.py && \
    echo '    if len(sys.argv) > 1 and sys.argv[1] == "verify":' >> /app/mlflow_manager.py && \
    echo '        success = verify_model_registration()' >> /app/mlflow_manager.py && \
    echo '        sys.exit(0 if success else 1)' >> /app/mlflow_manager.py && \
    echo '    mlflow_proc = start_mlflow_server()' >> /app/mlflow_manager.py && \
    echo '    def handle_signal(sig, frame):' >> /app/mlflow_manager.py && \
    echo '        if sig in (signal.SIGINT, signal.SIGTERM):' >> /app/mlflow_manager.py && \
    echo '            print("Sinal de terminação recebido, mas mantendo MLflow rodando para o pipeline")' >> /app/mlflow_manager.py && \
    echo '        else:' >> /app/mlflow_manager.py && \
    echo '            print(f"Sinal recebido {sig}")' >> /app/mlflow_manager.py && \
    echo '    signal.signal(signal.SIGINT, handle_signal)' >> /app/mlflow_manager.py && \
    echo '    signal.signal(signal.SIGTERM, handle_signal)' >> /app/mlflow_manager.py && \
    echo '    try:' >> /app/mlflow_manager.py && \
    echo '        while mlflow_proc.poll() is None:' >> /app/mlflow_manager.py && \
    echo '            time.sleep(1)' >> /app/mlflow_manager.py && \
    echo '        print(f"MLflow encerrou com código {mlflow_proc.returncode}")' >> /app/mlflow_manager.py && \
    echo '        with open("mlflow.log", "r") as f:' >> /app/mlflow_manager.py && \
    echo '            print(f.read())' >> /app/mlflow_manager.py && \
    echo '    except KeyboardInterrupt:' >> /app/mlflow_manager.py && \
    echo '        print("Interrompido pelo usuário")' >> /app/mlflow_manager.py && \
    echo '        mlflow_proc.terminate()' >> /app/mlflow_manager.py

# Cria o script para fixar o comportamento de mlflow.end_run()
RUN echo 'import re' > /app/fix_pipeline.py && \
    echo 'import os' >> /app/fix_pipeline.py && \
    echo 'def fix_mlflow_pattern():' >> /app/fix_pipeline.py && \
    echo '    """Corrige manualmente padrões específicos no código."""' >> /app/fix_pipeline.py && \
    echo '    # Arquivos a serem modificados' >> /app/fix_pipeline.py && \
    echo '    files = ["pipeline.py", "src/modelo/recomendador.py", "src/config/mlflow_config.py"]' >> /app/fix_pipeline.py && \
    echo '    for file in files:' >> /app/fix_pipeline.py && \
    echo '        if not os.path.exists(file):' >> /app/fix_pipeline.py && \
    echo '            print(f"Arquivo {file} não encontrado")' >> /app/fix_pipeline.py && \
    echo '            continue' >> /app/fix_pipeline.py && \
    echo '        # Lê o conteúdo do arquivo' >> /app/fix_pipeline.py && \
    echo '        with open(file, "r") as f:' >> /app/fix_pipeline.py && \
    echo '            content = f.read()' >> /app/fix_pipeline.py && \
    echo '        # Adiciona import do MLflow_TRACKING_URI se necessário' >> /app/fix_pipeline.py && \
    echo '        if "import mlflow" in content and not "MLFLOW_TRACKING_URI" in content:' >> /app/fix_pipeline.py && \
    echo '            content = content.replace("import mlflow", "import mlflow\\nimport os\\nos.environ[\'MLFLOW_TRACKING_URI\'] = \'sqlite:///mlflow.db\'")' >> /app/fix_pipeline.py && \
    echo '        # Procura e corrige mlflow.end_run()' >> /app/fix_pipeline.py && \
    echo '        if "mlflow.end_run()" in content:' >> /app/fix_pipeline.py && \
    echo '            lines = content.split("\\n")' >> /app/fix_pipeline.py && \
    echo '            new_lines = []' >> /app/fix_pipeline.py && \
    echo '            for i, line in enumerate(lines):' >> /app/fix_pipeline.py && \
    echo '                if "mlflow.end_run()" in line:' >> /app/fix_pipeline.py && \
    echo '                    indent = len(line) - len(line.lstrip())' >> /app/fix_pipeline.py && \
    echo '                    indentation = " " * indent' >> /app/fix_pipeline.py && \
    echo '                    new_lines.append(f"{indentation}try:")' >> /app/fix_pipeline.py && \
    echo '                    new_lines.append(f"{indentation}    mlflow.end_run()")' >> /app/fix_pipeline.py && \
    echo '                    new_lines.append(f"{indentation}except Exception as e:")' >> /app/fix_pipeline.py && \
    echo '                    new_lines.append(f"{indentation}    print(f\\"Erro ao finalizar MLflow run: {{e}}\\")")' >> /app/fix_pipeline.py && \
    echo '                else:' >> /app/fix_pipeline.py && \
    echo '                    new_lines.append(line)' >> /app/fix_pipeline.py && \
    echo '            content = "\\n".join(new_lines)' >> /app/fix_pipeline.py && \
    echo '        # Escreve o conteúdo modificado de volta ao arquivo' >> /app/fix_pipeline.py && \
    echo '        with open(file, "w") as f:' >> /app/fix_pipeline.py && \
    echo '            f.write(content)' >> /app/fix_pipeline.py && \
    echo '        print(f"Arquivo {file} corrigido")' >> /app/fix_pipeline.py && \
    echo '' >> /app/fix_pipeline.py && \
    echo 'if __name__ == "__main__":' >> /app/fix_pipeline.py && \
    echo '    fix_mlflow_pattern()' >> /app/fix_pipeline.py && \
    chmod +x /app/fix_pipeline.py

# Aplica as correções ao pipeline
RUN python /app/fix_pipeline.py 

# Cria entrypoint.sh usando echo linha por linha
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'echo "Verificando dependências..."' >> /app/entrypoint.sh && \
    echo 'pip list | grep -E "mlflow|flask|pyspark|werkzeug|nltk"' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Verifica instalações do Python e Java' >> /app/entrypoint.sh && \
    echo 'echo "Caminho do Python: $(which python)"' >> /app/entrypoint.sh && \
    echo 'echo "Versão do Python: $(python --version)"' >> /app/entrypoint.sh && \
    echo 'echo "Versão do Java:"' >> /app/entrypoint.sh && \
    echo 'java -version' >> /app/entrypoint.sh && \
    echo 'echo "JAVA_HOME=$JAVA_HOME"' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Testa configuração do PySpark' >> /app/entrypoint.sh && \
    echo 'echo "Testando configuração do PySpark..."' >> /app/entrypoint.sh && \
    echo 'python test_pyspark.py' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Garante versões corretas do Python para o PySpark' >> /app/entrypoint.sh && \
    echo 'export PYSPARK_PYTHON=$(which python3.10)' >> /app/entrypoint.sh && \
    echo 'export PYSPARK_DRIVER_PYTHON=$(which python3.10)' >> /app/entrypoint.sh && \
    echo 'echo "PYSPARK_PYTHON=$PYSPARK_PYTHON"' >> /app/entrypoint.sh && \
    echo 'echo "PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON"' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Inicializa banco de dados MLflow diretamente' >> /app/entrypoint.sh && \
    echo 'echo "Inicializando banco de dados MLflow..."' >> /app/entrypoint.sh && \
    echo 'python init_mlflow_db.py' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Inicia MLflow de forma a garantir que permaneça rodando durante a execução do pipeline' >> /app/entrypoint.sh && \
    echo 'echo "Iniciando MLflow persistente..."' >> /app/entrypoint.sh && \
    echo 'python mlflow_manager.py &' >> /app/entrypoint.sh && \
    echo 'MLFLOW_PID=$!' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Aguarda o MLflow estar pronto' >> /app/entrypoint.sh && \
    echo 'echo "Aguardando MLflow iniciar..."' >> /app/entrypoint.sh && \
    echo 'max_attempts=30' >> /app/entrypoint.sh && \
    echo 'attempt=0' >> /app/entrypoint.sh && \
    echo 'mlflow_ready=false' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'while [ $attempt -lt $max_attempts ]; do' >> /app/entrypoint.sh && \
    echo '    attempt=$((attempt+1))' >> /app/entrypoint.sh && \
    echo '    echo "Verificando MLflow (tentativa $attempt/$max_attempts)..."' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    if curl -s http://localhost:5000/ > /dev/null; then' >> /app/entrypoint.sh && \
    echo '        echo "MLflow iniciado com sucesso!"' >> /app/entrypoint.sh && \
    echo '        mlflow_ready=true' >> /app/entrypoint.sh && \
    echo '        break' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Verifica se o processo MLflow ainda está rodando' >> /app/entrypoint.sh && \
    echo '    if ! kill -0 $MLFLOW_PID 2>/dev/null; then' >> /app/entrypoint.sh && \
    echo '        echo "ERRO: Processo do MLflow encerrou prematuramente."' >> /app/entrypoint.sh && \
    echo '        cat mlflow.log 2>/dev/null || echo "Log do MLflow não encontrado"' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    sleep 2' >> /app/entrypoint.sh && \
    echo 'done' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'if [ "$mlflow_ready" = false ]; then' >> /app/entrypoint.sh && \
    echo '    echo "ERRO: MLflow não iniciou corretamente após várias tentativas."' >> /app/entrypoint.sh && \
    echo '    cat mlflow.log 2>/dev/null || echo "Log do MLflow não encontrado"' >> /app/entrypoint.sh && \
    echo '    exit 1' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Se solicitado, executa o pipeline primeiro' >> /app/entrypoint.sh && \
    echo 'if [ "$RUN_PIPELINE" = "true" ]; then' >> /app/entrypoint.sh && \
    echo '    echo "Executando o pipeline..."' >> /app/entrypoint.sh && \
    echo '    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \\' >> /app/entrypoint.sh && \
    echo '    PYSPARK_PYTHON=$PYSPARK_PYTHON \\' >> /app/entrypoint.sh && \
    echo '    PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON \\' >> /app/entrypoint.sh && \
    echo '    python pipeline.py' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Verifica status de saída do pipeline' >> /app/entrypoint.sh && \
    echo '    PIPELINE_STATUS=$?' >> /app/entrypoint.sh && \
    echo '    if [ $PIPELINE_STATUS -ne 0 ]; then' >> /app/entrypoint.sh && \
    echo '        echo "ERRO: Pipeline falhou com código de saída $PIPELINE_STATUS"' >> /app/entrypoint.sh && \
    echo '        exit $PIPELINE_STATUS' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Verifica registro de modelos no MLflow' >> /app/entrypoint.sh && \
    echo '    echo "Verificando registro de modelos no MLflow..."' >> /app/entrypoint.sh && \
    echo '    python mlflow_manager.py verify' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Verifica se os modelos foram criados' >> /app/entrypoint.sh && \
    echo '    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then' >> /app/entrypoint.sh && \
    echo '        echo "AVISO: Nenhum modelo foi encontrado após a execução do pipeline."' >> /app/entrypoint.sh && \
    echo '        echo "A API não será iniciada sem modelos treinados."' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '    # Verifica se os modelos existem' >> /app/entrypoint.sh && \
    echo '    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then' >> /app/entrypoint.sh && \
    echo '        echo "ERRO: Nenhum modelo encontrado em modelos/modelos_salvos/."' >> /app/entrypoint.sh && \
    echo '        echo "Execute o contêiner com -e RUN_PIPELINE=true para treinar os modelos primeiro"' >> /app/entrypoint.sh && \
    echo '        echo "ou monte um volume com modelos pré-treinados."' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Função para lidar com o encerramento do contêiner de forma elegante' >> /app/entrypoint.sh && \
    echo 'cleanup() {' >> /app/entrypoint.sh && \
    echo '    echo "Recebido sinal de encerramento. Encerrando processos..."' >> /app/entrypoint.sh && \
    echo '    if [ -n "$API_PID" ]; then' >> /app/entrypoint.sh && \
    echo '        echo "Encerrando API (PID: $API_PID)..."' >> /app/entrypoint.sh && \
    echo '        kill -TERM $API_PID 2>/dev/null || true' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    if [ -n "$MLFLOW_PID" ]; then' >> /app/entrypoint.sh && \
    echo '        echo "Encerrando MLflow (PID: $MLFLOW_PID)..."' >> /app/entrypoint.sh && \
    echo '        kill -TERM $MLFLOW_PID 2>/dev/null || true' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    echo "Processos encerrados, saindo..."' >> /app/entrypoint.sh && \
    echo '    exit 0' >> /app/entrypoint.sh && \
    echo '}' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Configura trap para cleanup ao parar o contêiner' >> /app/entrypoint.sh && \
    echo 'trap cleanup SIGTERM SIGINT' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Arquivo para monitoramento de logs da API' >> /app/entrypoint.sh && \
    echo 'touch api.log' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Inicia o servidor da API em segundo plano' >> /app/entrypoint.sh && \
    echo 'echo "Iniciando API..."' >> /app/entrypoint.sh && \
    echo './scripts/start_api.sh > api.log 2>&1 &' >> /app/entrypoint.sh && \
    echo 'API_PID=$!' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Aguarda a API iniciar' >> /app/entrypoint.sh && \
    echo 'echo "Aguardando API iniciar..."' >> /app/entrypoint.sh && \
    echo 'sleep 5' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Verifica se a API está rodando' >> /app/entrypoint.sh && \
    echo 'if ! kill -0 $API_PID 2>/dev/null; then' >> /app/entrypoint.sh && \
    echo '    echo "ERRO: API falhou ao iniciar. Verificando logs:"' >> /app/entrypoint.sh && \
    echo '    cat api.log' >> /app/entrypoint.sh && \
    echo '    exit 1' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'echo "==============================================="' >> /app/entrypoint.sh && \
    echo 'echo "MLflow UI está rodando em: http://localhost:5000"' >> /app/entrypoint.sh && \
    echo 'echo "API está rodando em: http://localhost:8000"' >> /app/entrypoint.sh && \
    echo 'echo "==============================================="' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Monitora ambos os processos e mantém o contêiner rodando' >> /app/entrypoint.sh && \
    echo 'echo "Monitorando processos..."' >> /app/entrypoint.sh && \
    echo 'while true; do' >> /app/entrypoint.sh && \
    echo '    # Verifica se o MLflow ainda está rodando' >> /app/entrypoint.sh && \
    echo '    if ! kill -0 $MLFLOW_PID 2>/dev/null; then' >> /app/entrypoint.sh && \
    echo '        echo "ALERTA: MLflow encerrou inesperadamente. Tentando reiniciar..."' >> /app/entrypoint.sh && \
    echo '        python mlflow_manager.py &' >> /app/entrypoint.sh && \
    echo '        MLFLOW_PID=$!' >> /app/entrypoint.sh && \
    echo '        echo "MLflow reiniciado com PID: $MLFLOW_PID"' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Verifica se a API ainda está rodando' >> /app/entrypoint.sh && \
    echo '    if ! kill -0 $API_PID 2>/dev/null; then' >> /app/entrypoint.sh && \
    echo '        echo "ALERTA: API encerrou inesperadamente. Tentando reiniciar..."' >> /app/entrypoint.sh && \
    echo '        ./scripts/start_api.sh > api.log 2>&1 &' >> /app/entrypoint.sh && \
    echo '        API_PID=$!' >> /app/entrypoint.sh && \
    echo '        echo "API reiniciada com PID: $API_PID"' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Se ambos os processos falharem, encerra o contêiner' >> /app/entrypoint.sh && \
    echo '    if ! kill -0 $MLFLOW_PID 2>/dev/null && ! kill -0 $API_PID 2>/dev/null; then' >> /app/entrypoint.sh && \
    echo '        echo "ERRO: Tanto o MLflow quanto a API falharam múltiplas vezes. Encerrando contêiner."' >> /app/entrypoint.sh && \
    echo '        echo "Verificando logs do MLflow:"' >> /app/entrypoint.sh && \
    echo '        cat mlflow.log 2>/dev/null || echo "Log do MLflow não encontrado"' >> /app/entrypoint.sh && \
    echo '        echo "Verificando logs da API:"' >> /app/entrypoint.sh && \
    echo '        cat api.log' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo '    ' >> /app/entrypoint.sh && \
    echo '    # Dorme e verifica novamente' >> /app/entrypoint.sh && \
    echo '    sleep 10' >> /app/entrypoint.sh && \
    echo 'done' >> /app/entrypoint.sh

# Torna o entrypoint executável
RUN chmod +x /app/entrypoint.sh

# Define o entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]