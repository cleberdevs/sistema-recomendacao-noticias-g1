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
    echo "Executando o pipeline com wrapper MLflow..."
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    PYSPARK_PYTHON=$PYSPARK_PYTHON \
    PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON \
    python mlflow_wrapper.py pipeline.py
    
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