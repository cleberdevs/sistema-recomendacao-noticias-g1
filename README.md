# Sistema de Recomendação Híbrido de Notícias com MLflow

Sistema de recomendação que combina abordagens de filtragem colaborativa, baseada em conteúdo e temporal, com rastreamento de experimentos usando MLflow.

## Estrutura do Projeto
```
projeto/
│
├── src/
│   ├── modelo/
│   │   ├── recomendador.py
│   │   └── preprocessamento.py
│   ├── api/
│   │   └── app.py
│   ├── config/
│   │   └── mlflow_config.py
│   └── utils/
│       └── helpers.py
│
├── dados/
│   ├── brutos/
│   └── processados/
│
├── modelos/
│   └── modelos_salvos/
│
├── testes/
│   ├── teste_modelo.py
│   └── teste_api.py
│
├── mlflow-artifacts/
└── logs/
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

### 1. Treinamento do Modelo

```bash
python treinar.py
```

Este comando irá:
- Processar os dados brutos
- Treinar o modelo híbrido
- Registrar métricas e parâmetros no MLflow
- Salvar o modelo treinado

### 2. Iniciar Serviços

```bash
./start_services.sh
```

Este script inicia:
- Servidor MLflow na porta 5000
- API Flask na porta 8000

### 3. MLflow UI

Acesse: http://localhost:5000

Funcionalidades:
- Visualização de experimentos
- Comparação de modelos
- Métricas e parâmetros
- Artefatos

### 4. API Endpoints

#### Verificação de Saúde
```bash
GET /saude
```

#### Obter Recomendações
```bash
POST /prever
Content-Type: application/json

{
    "id_usuario": "string",
    "n_recomendacoes": integer
}
```

### 5. Docker

Construir imagem:
```bash
docker build -t recomendador-noticias .
```

Executar container:
```bash
docker run -p 8000:8000 -p 5000:5000 recomendador-noticias
```

### 6. Testes

Executar testes:
```bash
python -m pytest testes/
```

## MLflow

### Experimentos
- Cada execução de treino é registrada como um experimento
- Parâmetros e métricas são registrados automaticamente
- Modelos são salvos como artefatos

### Métricas Registradas
- Acurácia do modelo
- Loss do treinamento
- Métricas de recomendação (precisão, recall)
- Métricas de dados (número de usuários, itens)

### Parâmetros Rastreados
- Dimensões dos embeddings
- Tamanho das features de texto
- Parâmetros de treinamento
- Configurações do modelo

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Crie um Pull Request

