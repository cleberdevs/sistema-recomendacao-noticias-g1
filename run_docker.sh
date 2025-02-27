#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Função para log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

# Função para verificar resultado do último comando
check_result() {
    if [ $? -ne 0 ]; then
        error "$1"
        exit 1
    fi
}

# Nome da imagem e container
IMAGE_NAME="recomendador-noticias"
CONTAINER_NAME="recomendador"

# Verificar se está rodando como root ou com sudo
if [ "$EUID" -ne 0 ]; then 
    error "Por favor, execute este script como root ou com sudo"
    exit 1
fi

# Verificar se Docker está instalado e rodando
if ! command -v docker &> /dev/null; then
    error "Docker não está instalado"
    exit 1
fi

if ! docker info &> /dev/null; then
    error "Docker daemon não está rodando"
    exit 1
fi

# Limpar ambiente anterior
log "Limpando ambiente anterior..."
docker stop $CONTAINER_NAME &>/dev/null || true
docker rm $CONTAINER_NAME &>/dev/null || true
docker rmi $IMAGE_NAME &>/dev/null || true
rm -rf mlflow.db mlruns/* 2>/dev/null || true

# Criar estrutura de diretórios
log "Criando estrutura de diretórios..."
mkdir -p \
    dados/brutos/itens \
    dados/processados \
    logs \
    mlflow-artifacts \
    modelos/modelos_salvos \
    checkpoints \
    spark-logs

# Configurar permissões
log "Configurando permissões..."
chmod -R 777 \
    dados \
    logs \
    mlflow-artifacts \
    modelos \
    checkpoints \
    spark-logs
chmod +x scripts/*.sh docker-entrypoint.sh
check_result "Falha ao configurar permissões"

# Verificar se tem dados brutos
if [ -z "$(ls -A dados/brutos)" ]; then
    warn "Diretório de dados brutos está vazio"
    warn "Certifique-se de adicionar os arquivos de dados em:"
    warn "  - dados/brutos/treino_parte*.csv"
    warn "  - dados/brutos/itens/itens-parte*.csv"
    read -p "Deseja continuar mesmo assim? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Construir imagem
log "Construindo imagem Docker..."
docker build -t $IMAGE_NAME . --no-cache
check_result "Falha ao construir imagem"

# Iniciar container
log "Iniciando container..."
docker run -d \
    --name $CONTAINER_NAME \
    --network host \
    -p 8000:8000 \
    -p 5000:5000 \
    -v $(pwd)/dados:/app/dados \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/mlflow-artifacts:/app/mlflow-artifacts \
    -v $(pwd)/mlruns:/app/mlruns \
    $IMAGE_NAME
check_result "Falha ao iniciar container"

# Função para verificar serviço
check_service() {
    local service=$1
    local port=$2
    local endpoint=$3
    local max_attempts=$4
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log "Verificando $service (tentativa $attempt de $max_attempts)..."
        
        if [ "$service" = "API" ]; then
            # Para a API, verificar se o modelo está carregado
            response=$(curl -s http://localhost:$port$endpoint)
            if echo "$response" | grep -q "saudavel"; then
                log "$service está rodando com modelo carregado"
                return 0
            fi
        else
            # Para outros serviços (MLflow)
            if curl -s http://localhost:$port/health &> /dev/null; then
                log "$service está rodando"
                return 0
            fi
        fi
        
        # Se falhou, verificar logs
        if [ $attempt -eq $max_attempts ]; then
            error "$service não iniciou após $max_attempts tentativas"
            error "=== Logs do Container ==="
            docker logs $CONTAINER_NAME
            error "=== Fim dos Logs ==="
            return 1
        fi
        
        log "Aguardando $service iniciar... ($attempt/$max_attempts)"
        attempt=$((attempt + 1))
        sleep 5
    done
}

# Aguardar serviços iniciarem
log "Aguardando serviços iniciarem..."
sleep 10

# Verificar MLflow (60 tentativas = 5 minutos)
if ! check_service "MLflow" 5000 "/health" 6; then
    error "Problema ao iniciar MLflow. Tentando reiniciar o container..."
    docker restart $CONTAINER_NAME
    sleep 10
    if ! check_service "MLflow" 5000 "/health" 3; then
        error "Falha persistente no MLflow. Encerrando..."
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    fi
fi

# Verificar se o modelo está sendo treinado
log "Verificando status do treinamento..."
while true; do
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Treinamento concluído com sucesso"; then
        log "Treinamento do modelo concluído com sucesso"
        break
    elif docker logs $CONTAINER_NAME 2>&1 | grep -q "Modelo existente verificado com sucesso"; then
        log "Modelo existente verificado com sucesso"
        break
    elif docker logs $CONTAINER_NAME 2>&1 | grep -q "ERRO: Falha no treinamento do modelo"; then
        error "Falha no treinamento do modelo"
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    fi
    log "Aguardando treinamento..."
    sleep 10
done

# Verificar API (30 tentativas = 2.5 minutos)
if ! check_service "API" 8000 "/api/sistema/saude" 30; then
    error "API não iniciou corretamente ou modelo não está carregado"
    docker logs $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
fi

# Mostrar informações finais
log "\nServiços disponíveis:"
echo -e "${GREEN}MLflow UI:${NC} http://localhost:5000"
echo -e "${GREEN}API Swagger:${NC} http://localhost:8000/docs"
echo -e "${GREEN}Interface Web:${NC} http://localhost:8000"

log "\nComandos úteis:"
echo -e "Ver logs: ${YELLOW}docker logs -f $CONTAINER_NAME${NC}"
echo -e "Parar container: ${YELLOW}docker stop $CONTAINER_NAME${NC}"
echo -e "Remover container: ${YELLOW}docker rm $CONTAINER_NAME${NC}"
echo -e "Reiniciar container: ${YELLOW}docker restart $CONTAINER_NAME${NC}"

# Mostrar status final dos serviços
log "\nVerificação final dos serviços:"
if curl -s http://localhost:5000/health > /dev/null; then
    echo -e "${GREEN}MLflow: OK${NC}"
else
    echo -e "${RED}MLflow: FALHA${NC}"
fi

if curl -s http://localhost:8000/api/sistema/saude > /dev/null; then
    echo -e "${GREEN}API: OK${NC}"
else
    echo -e "${RED}API: FALHA${NC}"
fi

# Verificar se há erros nos logs
if docker logs $CONTAINER_NAME 2>&1 | grep -i "error" > /dev/null; then
    warn "\nEncontrados possíveis erros nos logs. Verifique com:"
    echo -e "${YELLOW}docker logs $CONTAINER_NAME | grep -i error${NC}"
fi

log "\nDeploy concluído!"