# Cores para output
$GREEN = [System.ConsoleColor]::Green
$RED = [System.ConsoleColor]::Red
$YELLOW = [System.ConsoleColor]::Yellow

# Função para log
function Write-Log {
    param($Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor $GREEN
}

function Write-Error-Log {
    param($Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: $Message" -ForegroundColor $RED
}

function Write-Warning-Log {
    param($Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WARN: $Message" -ForegroundColor $YELLOW
}

# Função para verificar resultado do último comando
function Check-Result {
    param($Message)
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Log $Message
        exit 1
    }
}

# Nome da imagem e container
$IMAGE_NAME = "recomendador-noticias"
$CONTAINER_NAME = "recomendador"

# Verificar se Docker está instalado e rodando
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error-Log "Docker não está instalado"
    exit 1
}

try {
    docker info | Out-Null
} catch {
    Write-Error-Log "Docker daemon não está rodando"
    exit 1
}

# Limpar ambiente anterior
Write-Log "Limpando ambiente anterior..."
docker stop $CONTAINER_NAME 2>$null
docker rm $CONTAINER_NAME 2>$null
docker rmi $IMAGE_NAME 2>$null
Remove-Item -Path "mlflow.db", "mlruns/*" -Force -ErrorAction SilentlyContinue

# Criar estrutura de diretórios
Write-Log "Criando estrutura de diretórios..."
$dirs = @(
    "dados\brutos\itens",
    "dados\processados",
    "logs",
    "mlflow-artifacts",
    "modelos\modelos_salvos",
    "checkpoints",
    "spark-logs"
)

foreach ($dir in $dirs) {
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
}

# Verificar se tem dados brutos
if (-not (Get-ChildItem -Path "dados\brutos" -File)) {
    Write-Warning-Log "Diretório de dados brutos está vazio"
    Write-Warning-Log "Certifique-se de adicionar os arquivos de dados em:"
    Write-Warning-Log "  - dados\brutos\treino_parte*.csv"
    Write-Warning-Log "  - dados\brutos\itens\itens-parte*.csv"
    $response = Read-Host "Deseja continuar mesmo assim? (y/n)"
    if ($response -notmatch "[yY]") {
        exit 1
    }
}

# Construir imagem
Write-Log "Construindo imagem Docker..."
docker build -t $IMAGE_NAME . --no-cache
Check-Result "Falha ao construir imagem"

# Iniciar container
Write-Log "Iniciando container..."
$current_path = (Get-Location).Path
docker run -d `
    --name $CONTAINER_NAME `
    -p 8000:8000 `
    -p 5000:5000 `
    -v "${current_path}\dados:/app/dados" `
    -v "${current_path}\logs:/app/logs" `
    -v "${current_path}\mlflow-artifacts:/app/mlflow-artifacts" `
    -v "${current_path}\mlruns:/app/mlruns" `
    $IMAGE_NAME
Check-Result "Falha ao iniciar container"

# Função para verificar serviço
function Check-Service {
    param(
        $ServiceName,
        $Port,
        $Endpoint,
        $MaxAttempts
    )
    
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        Write-Log "Verificando $ServiceName (tentativa $attempt de $MaxAttempts)..."
        
        try {
            if ($ServiceName -eq "API") {
                $response = Invoke-RestMethod -Uri "http://localhost:$Port$Endpoint" -Method Get
                if ($response.status -eq "saudavel") {
                    Write-Log "$ServiceName está rodando com modelo carregado"
                    return $true
                }
            } else {
                $response = Invoke-RestMethod -Uri "http://localhost:$Port/health" -Method Get
                Write-Log "$ServiceName está rodando"
                return $true
            }
        } catch {
            if ($attempt -eq $MaxAttempts) {
                Write-Error-Log "$ServiceName não iniciou após $MaxAttempts tentativas"
                Write-Error-Log "=== Logs do Container ==="
                docker logs $CONTAINER_NAME
                Write-Error-Log "=== Fim dos Logs ==="
                return $false
            }
        }
        
        Write-Log "Aguardando $ServiceName iniciar... ($attempt/$MaxAttempts)"
        $attempt++
        Start-Sleep -Seconds 5
    }
    return $false
}

# Aguardar serviços iniciarem
Write-Log "Aguardando serviços iniciarem..."
Start-Sleep -Seconds 10

# Verificar MLflow
if (-not (Check-Service -ServiceName "MLflow" -Port 5000 -Endpoint "/health" -MaxAttempts 6)) {
    Write-Error-Log "Problema ao iniciar MLflow. Tentando reiniciar o container..."
    docker restart $CONTAINER_NAME
    Start-Sleep -Seconds 10
    if (-not (Check-Service -ServiceName "MLflow" -Port 5000 -Endpoint "/health" -MaxAttempts 3)) {
        Write-Error-Log "Falha persistente no MLflow. Encerrando..."
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    }
}

# Verificar se o modelo está sendo treinado
Write-Log "Verificando status do treinamento..."
$training_complete = $false
while (-not $training_complete) {
    $logs = docker logs $CONTAINER_NAME 2>&1
    if ($logs -match "Treinamento concluído com sucesso") {
        Write-Log "Treinamento do modelo concluído com sucesso"
        $training_complete = $true
    } elseif ($logs -match "Modelo existente verificado com sucesso") {
        Write-Log "Modelo existente verificado com sucesso"
        $training_complete = $true
    } elseif ($logs -match "ERRO: Falha no treinamento do modelo") {
        Write-Error-Log "Falha no treinamento do modelo"
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    }
    if (-not $training_complete) {
        Write-Log "Aguardando treinamento..."
        Start-Sleep -Seconds 10
    }
}

# Verificar API
if (-not (Check-Service -ServiceName "API" -Port 8000 -Endpoint "/api/sistema/saude" -MaxAttempts 30)) {
    Write-Error-Log "API não iniciou corretamente ou modelo não está carregado"
    docker logs $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
}

# Mostrar informações finais
Write-Log "`nServiços disponíveis:"
Write-Host "MLflow UI: http://localhost:5000" -ForegroundColor $GREEN
Write-Host "API Swagger: http://localhost:8000/docs" -ForegroundColor $GREEN
Write-Host "Interface Web: http://localhost:8000" -ForegroundColor $GREEN

Write-Log "`nComandos úteis:"
Write-Host "Ver logs: docker logs -f $CONTAINER_NAME" -ForegroundColor $YELLOW
Write-Host "Parar container: docker stop $CONTAINER_NAME" -ForegroundColor $YELLOW
Write-Host "Remover container: docker rm $CONTAINER_NAME" -ForegroundColor $YELLOW
Write-Host "Reiniciar container: docker restart $CONTAINER_NAME" -ForegroundColor $YELLOW

Write-Log "`nDeploy concluído!"