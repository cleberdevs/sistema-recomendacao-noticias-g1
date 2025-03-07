
# Sistema de Recomendação de Notícias do Portal G1

Sistema de recomendação híbrido que combina filtragem colaborativa, análise de conteúdo, fatores temporais e de popularidade para gerar recomendações personalizadas de notícias.

## Sobre o Projeto

### Autoria
**Grupo 60 - Pós Tech FIAP - Engenharia de Machine Learning**

### Contexto Acadêmico
Este projeto foi desenvolvido como parte do curso de Pós-Graduação em Engenharia de Machine Learning da FIAP (Faculdade de Informática e Administração Paulista), representando a aplicação prática dos conhecimentos adquiridos na Fase 5 - MLOPS.


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
- Docker


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
source venv/bin/activate  
```

### 3. Torne os scripts executáveis (Linux/Mac):
```bash
chmod +x scripts/start_mlflow.sh
chmod +x scripts/start_api.sh
chmod +x scripts/healthcheck.sh
chmod +x scripts/setup_environment.sh
```

### 4. Configure o ambiente:
```bash
./scripts/setup_environment.sh
```
### 5.📁 Localização dos Dados Brutos

Os dados brutos devem ser colocados nos seguintes diretórios:


```
dados/
├── brutos/
│   ├── treino_parte1.csv
│   ├── treino_parte2.csv
│   └── itens/
│       ├── itens-parte1.csv
│       └── itens-parte2.csv
```



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

### 4. Acesse a interface web da API:
```
http://localhost:8000
```

A documentação Swagger estará disponível localmente em:
```
http://localhost:8000/docs
```

## ⚙️ Instalação Docker

Estas instruções permitirão que você execute uma cópia do projeto em sua máquina local ou na nuvem para produção.

### 1. Clone o repositório:
```bash
git clone https://github.com/cleberdevs/sistema-recomendacao-noticias-g1.git
cd sistema-recomendacao-noticias-g1
```
### 2. Crie os diretórios necessários:
```bash
mkdir -p dados/brutos/itens logs modelos/modelos_salvos
```
### 3.📁 Localização dos Dados Brutos

Os dados brutos devem ser colocados nos seguintes diretórios:

```
dados/
├── brutos/
│   ├── treino_parte1.csv
│   ├── treino_parte2.csv
│   └── itens/
│       ├── itens-parte1.csv
│       └── itens-parte2.csv
```
### 4. Crie a imagem:
```bash
docker build -t sistema-recomendacao-g1 .
```

## 🚀 Execução Docker - pipeline não executado

### 1. Execute o comando para criar o container:
```bash
docker run -p 5000:5000 -p 8000:8000 -e RUN_PIPELINE=true -v $(pwd)/dados:/app/dados -v $(pwd)/logs:/app/logs -v $(pwd)/modelos/modelos_salvos:/app/modelos/modelos_salvos --name recomendador sistema-recomendacao-g1 
```
### 2. Acesse a interface web da API:
```
http://localhost:8000
```

A documentação Swagger estará disponível localmente em:
```
http://localhost:8000/docs
```

## 🚀 Execução Docker - pipeline executado com dados processados e modelo treinado

##  **Paraesta etapa é preciso apenas executar etapa 1 da Instalação Docker**

### 1. Execute o comando para criar o container:
```bash
docker run -p 5000:5000 -p 8000:8000 -e RUN_PIPELINE=false --name recomendador cleberfx/sistema-recomendacao-g1:dados-processados-modelo-treinado 
```

### 2. Acesse a interface web da API:
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

### 8. Detalhamento da Arquitetura Neural

O sistema implementa uma arquitetura neural profunda que combina filtragem colaborativa e baseada em conteúdo, composta por 11 camadas principais organizadas em 6 grupos funcionais:

#### 8.1 Camadas de Entrada (3 camadas)
O modelo recebe três tipos de entrada:
- Entrada de Usuários: 1 neurônio (ID do usuário)
- Entrada de Itens: 1 neurônio (ID do item)
- Entrada de Conteúdo: 100 neurônios (vetor TF-IDF dos textos)

#### 8.2 Camadas de Embedding - Filtragem Colaborativa (2 camadas)
Implementa a filtragem colaborativa através de:
- Embedding de Usuários: 16 neurônios
- Embedding de Itens: 16 neurônios
Características:
- Transforma IDs em vetores densos
- Captura padrões latentes de interação
- Aprende representações automáticas de preferências

#### 8.3 Camadas de Flatten (2 camadas)
Processamento dos embeddings:
- Flatten Usuários: 16 neurônios
- Flatten Itens: 16 neurônios

#### 8.4 Camada de Concatenação (1 camada)
Fusão das diferentes abordagens:
- 132 neurônios totais (16 + 16 + 100)
- Unifica embeddings colaborativos
- Integra features de conteúdo
- Permite aprendizado conjunto dos padrões

#### 8.5 Camadas Densas com Regularização (2 blocos)
Processamento profundo estruturado em:

Primeiro Bloco:
- Dense: 32 neurônios
- Layer Normalization: 32 neurônios
- Dropout: 30% dos neurônios

Segundo Bloco:
- Dense: 16 neurônios
- Layer Normalization: 16 neurônios
- Dropout: 30% dos neurônios

#### 8.6 Camada de Saída (1 camada)
- 1 neurônio com ativação sigmoid
- Produz score entre 0 e 1
- Representa probabilidade de recomendação

#### 8.7 Fluxo de Dados
1 → 16 → 16 → 132 → 32 → 16 → 1
(entrada → embedding → flatten → concatenação → dense1 → dense2 → saída)

#### 8.8 Estratégias de Regularização
- Regularização L2: Controle de complexidade
- Dropout: Redução de overfitting (30%)
- Layer Normalization: Estabilidade no treinamento
- Gradient Clipping: Previne explosão do gradiente

#### 8.9 Vantagens da Arquitetura
1. **Flexibilidade**
   - Combina múltiplas fontes de informação
   - Adaptável a diferentes tipos de conteúdo

2. **Performance**
   - Embeddings eficientes
   - Regularização robusta
   - Treinamento estável

3. **Escalabilidade**
   - Processamento em batch
   - Arquitetura modular
   - Otimização de memória