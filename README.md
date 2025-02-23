
# Sistema de Recomendação de Notícias do Portal G1

Sistema de recomendação híbrido que combina filtragem colaborativa, análise de conteúdo, fatores temporais e de popularidade para gerar recomendações personalizadas de notícias.

## 🚀 Funcionalidades

- Recomendações personalizadas por usuário
- Tratamento de cold start para novos usuários 
- Interface web para visualização das recomendações
- API REST para integração
- Documentação Swagger
- Sistema de logs e monitoramento
- Integração com MLflow para tracking de experimentos
- Processamento distribuído com PySpark

## 🛠️ Tecnologias

- Python 3.8+
- TensorFlow 2.5+
- PySpark 3.2+
- Flask / Flask-RESTX
- MLflow
- HTML/CSS/JavaScript
- Bootstrap

## 📋 Pré-requisitos

- Python 3.8+
- Java 8+ 
- Pip
- Virtualenv (recomendado)



## ⚙️ Instalação Local

Estas instruções permitirão que você execute uma cópia do projeto em sua máquina local para fins de desenvolvimento e teste.

### 1. Clone o repositório:
```bash
git clone https://github.com/cleberdevs/sistema-recomendacao-noticias-g1.git
cd sistema-recomendacao-noticias-g1
```

### 2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Torne os scripts executáveis (Linux/Mac):
```bash
chmod +x scripts/start_mlflow.sh
chmod +x scripts/start_api.sh
chmod +x scripts/healthcheck.sh
chmod +x scripts/setup_environment.sh
```

### 3. Configure o ambiente:
```bash
./scripts/setup_environment.sh
```

Nota: Para usuários Windows, os scripts .sh não são necessários. Use os comandos Python diretamente ou crie scripts .bat equivalentes.

## 🚀 Execução Local

### 1. Inicie o servidor MLflow:
```bash
./scripts/start_mlflow.sh  
```

### 2. Inicie o pipeline:
```bash
python pipeline.py
```

### 3. Inicie a API:
```bash 
./scripts/start_api.sh  
```

### 4. Acesse a interface web local:
```
http://localhost:8000
```

A documentação Swagger estará disponível localmente em:
```
http://localhost:8000/docs
```



## 🏗️ Arquitetura

O sistema implementa uma arquitetura híbrida de recomendação que combina múltiplas estratégias em diferentes fases:

### 1. Modelo Neural Híbrido

O componente principal utiliza duas estratégias de filtragem implementadas em um modelo neural profundo:

- **Filtragem Colaborativa**: Implementada através de embeddings de usuários e itens que capturam padrões de interação
- **Filtragem Baseada em Conteúdo**: Implementada através de vetores TF-IDF do texto das notícias

### 2. Fase de Scoring e Recomendação

Após a construção e treino do modelo neural, são aplicadas estratégias adicionais:

#### 2.1 Para Usuários Existentes (Warm Start)
Utiliza todas as estratégias disponíveis com a seguinte ponderação:
- **60% Modelo Neural**: Combina filtragem colaborativa e baseada em conteúdo
- **25% Popularidade**: Score baseado no número de interações
- **15% Recência**: Peso temporal das notícias

#### 2.2 Para Novos Usuários (Cold Start)
Utiliza apenas estratégias independentes do histórico do usuário:
- **70% Popularidade**: Notícias mais acessadas
- **30% Recência**: Priorização de conteúdo atual

### 3. Pipeline de Processamento

```plaintext
Dados Brutos → Preprocessamento Spark → Features → Modelo → Recomendações → Serving
     ↓               ↓                     ↓          ↓          ↓             ↓
  CSV/JSON    Limpeza & Transformação    TF-IDF    Neural   Aplicação de    API REST
                                                            Estratégias
```


### 4. Componentes do Sistema

#### 4.1 Preprocessamento (PySpark)
- Limpeza de dados
- Normalização de timestamps
- Geração de features
- Processamento distribuído

#### 4.2 Modelo de Recomendação
- Framework: TensorFlow 2.x
- Arquitetura Neural: Filtragem Colaborativa e Análise de Conteúdo
- Otimizador: Adam
- Loss: Binary Crossentropy


#### 4.3 Estratégias Adicionais de Recomendação
- **Fatores Temporais**:
  - Pesos por período de tempo
  - Priorização de conteúdo recente
  - Decaimento temporal exponencial
- **Popularidade**:
  - Contagem de interações por item
  - Normalização por períodos
  - Score dinâmico atualizado

#### 4.4 Servidor de APIs (Flask)
- REST API
- Swagger UI
- Cache

#### 4.5 MLflow
- Tracking de experimentos
- Registro de modelos
- Monitoramento de métricas
- Versionamento

### 5. Monitoramento e Métricas

#### 5.1 Métricas do Modelo
- Accuracy
- Precision
- Recall
- Loss

#### 5.2 Métricas do Sistema
- Latência de resposta
- Taxa de acerto
- Diversidade das recomendações
- Cobertura do catálogo

### 6. Otimizações

- Batch predictions para eficiência
- Caching de recomendações frequentes
- Processamento assíncrono
- Load balancing
- Checkpointing

### 7. Segurança

- Validação de entrada
- Sanitização de dados
- Logs de auditoria
