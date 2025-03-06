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
python test_pyspark.py || echo "AVISO: Teste do PySpark falhou, mas continuando..."

# Garante versões corretas do Python para o PySpark
export PYSPARK_PYTHON=$(which python3.10)
export PYSPARK_DRIVER_PYTHON=$(which python3.10)
echo "PYSPARK_PYTHON=$PYSPARK_PYTHON"
echo "PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON"

# Inicializa banco de dados MLflow diretamente
echo "Inicializando banco de dados MLflow..."
# Corrige o erro de parâmetro faltante
python -c "
import os
import sys
import mlflow
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# Remove arquivo de banco de dados existente, se houver
# if os.path.exists('mlflow.db'):
    # os.remove('mlflow.db')

try:
    # Adiciona o parâmetro default_artifact_root
    store = SqlAlchemyStore('sqlite:///mlflow.db', './artifacts')
    store.get_experiment_by_name('Default')
    print('Banco de dados MLflow inicializado com sucesso')
except Exception as e:
    print(f'Erro ao inicializar banco de dados MLflow: {e}')
    print('Continuando mesmo com erro no banco de dados MLflow')
"

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
        echo "AVISO: Processo do MLflow encerrou prematuramente."
        cat mlflow.log
        echo "Tentando reiniciar o MLflow..."
        python mlflow_manager.py &
        MLFLOW_PID=$!
    fi
    
    sleep 2
done

if [ "$mlflow_ready" = false ]; then
    echo "AVISO: MLflow não iniciou corretamente após várias tentativas."
    cat mlflow.log
    echo "Continuando sem MLflow..."
fi

# Se solicitado, executa o pipeline primeiro
MODELS_EXIST=false
if [ "$RUN_PIPELINE" = "true" ]; then
    echo "Executando o pipeline..."
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    PYSPARK_PYTHON=$PYSPARK_PYTHON \
    PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON \
    python pipeline.py || echo "AVISO: Pipeline falhou, mas continuando..."
    
    # Verifica registro de modelos no MLflow (mas não falha se der erro)
    echo "Verificando registro de modelos no MLflow..."
    python mlflow_manager.py verify || echo "AVISO: Verificação de modelos falhou, mas continuando..."
    
    # Verifica se os modelos foram criados
    if [ "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then
        echo "Modelos encontrados após execução do pipeline."
        MODELS_EXIST=true
    else
        echo "AVISO: Nenhum modelo foi encontrado após a execução do pipeline."
        echo "A API não será iniciada sem modelos treinados."
        # Não sai, apenas marca que não há modelos
        MODELS_EXIST=false
    fi
else
    # Verifica se os modelos existem
    if [ "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then
        echo "Modelos encontrados no diretório modelos/modelos_salvos/."
        MODELS_EXIST=true
    else
        echo "AVISO: Nenhum modelo encontrado em modelos/modelos_salvos/."
        echo "Execute o contêiner com -e RUN_PIPELINE=true para treinar os modelos primeiro"
        echo "ou monte um volume com modelos pré-treinados."
        echo "A API não será iniciada, mas o contêiner continuará em execução."
        MODELS_EXIST=false
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

# Apenas inicia a API se houver modelos
if [ "$MODELS_EXIST" = true ]; then
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
        echo "AVISO: API falhou ao iniciar. Verificando logs:"
        cat api.log
        echo "O contêiner continuará em execução sem a API."
    else
        echo "API iniciada com sucesso na porta 8000"
    fi
fi

echo "==============================================="
if kill -0 $MLFLOW_PID 2>/dev/null; then
    echo "MLflow UI está rodando em: http://localhost:5000"
else
    echo "MLflow não está rodando (falha na inicialização)"
fi

if [ "$MODELS_EXIST" = true ] && [ -n "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
    echo "API está rodando em: http://localhost:8000"
else
    echo "API não está rodando (sem modelos ou falha na inicialização)"
fi
echo "==============================================="

# Monitora os processos e mantém o contêiner rodando
echo "Monitorando processos..."
while true; do
    # Verifica se o MLflow ainda está rodando e tenta reiniciar
    if [ -n "$MLFLOW_PID" ] && ! kill -0 $MLFLOW_PID 2>/dev/null; then
        echo "ALERTA: MLflow encerrou inesperadamente. Tentando reiniciar..."
        python mlflow_manager.py &
        NEW_MLFLOW_PID=$!
        
        sleep 5
        if kill -0 $NEW_MLFLOW_PID 2>/dev/null; then
            MLFLOW_PID=$NEW_MLFLOW_PID
            echo "MLflow reiniciado com PID: $MLFLOW_PID"
        else
            echo "Falha ao reiniciar MLflow."
            unset MLFLOW_PID
        fi
    fi
    
    # Verifica se a API ainda está rodando e tenta reiniciar
    if [ -n "$API_PID" ] && ! kill -0 $API_PID 2>/dev/null; then
        echo "ALERTA: API encerrou inesperadamente. Tentando reiniciar..."
        ./scripts/start_api.sh > api.log 2>&1 &
        NEW_API_PID=$!
        
        sleep 5
        if kill -0 $NEW_API_PID 2>/dev/null; then
            API_PID=$NEW_API_PID
            echo "API reiniciada com PID: $API_PID"
        else
            echo "Falha ao reiniciar API."
            unset API_PID
        fi
    fi
    
    # Imprime status periódico
    if [ $((SECONDS % 300)) -lt 10 ]; then
        echo "==============================================="
        echo "Status do contêiner - Ativo há $SECONDS segundos"
        if [ -n "$MLFLOW_PID" ] && kill -0 $MLFLOW_PID 2>/dev/null; then
            echo "MLflow rodando na porta 5000"
        else
            echo "MLflow não está rodando"
        fi
        
        if [ -n "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
            echo "API rodando na porta 8000"
        else
            echo "API não está rodando"
        fi
        echo "==============================================="
    fi
    
    # Dorme e verifica novamente - este loop é o que mantém o contêiner rodando
    sleep 10
done