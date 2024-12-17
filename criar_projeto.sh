#!/bin/bash

# criar_projeto.sh

# Cores para saída
VERMELHO='\033[0;31m'
VERDE='\033[0;32m'
SEM_COR='\033[0m'

echo -e "${VERDE}Criando estrutura do projeto de recomendação de notícias...${SEM_COR}"

# Criar estrutura de diretórios
mkdir -p projeto/{src/{modelo,api,utils},dados/{brutos,processados},modelos/modelos_salvos,testes,notebooks}

# Criar diretório para dados brutos
mkdir -p projeto/dados/brutos/itens

# Criar arquivos Python principais
cat > projeto/src/modelo/recomendador.py << 'EOL'
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
import pickle
from datetime import datetime

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100):
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        self.tfidf = TfidfVectorizer(max_features=dim_features_texto)
        self.itens_usuario = {}
        self.features_item = {}
        self.matriz_similaridade = None

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        # Camadas de entrada
        entrada_usuario = Input(shape=(1,))
        entrada_item = Input(shape=(1,))
        entrada_conteudo = Input(shape=(self.dim_features_texto,))

        # Embeddings de usuário/item
        embedding_usuario = Embedding(n_usuarios, self.dim_embedding)(entrada_usuario)
        embedding_item = Embedding(n_itens, self.dim_embedding)(entrada_item)

        # Achatar embeddings
        usuario_flat = Flatten()(embedding_usuario)
        item_flat = Flatten()(embedding_item)

        # Combinar features
        concat = Concatenate()([usuario_flat, item_flat, entrada_conteudo])

        # Camadas densas
        denso1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.3)(denso1)
        denso2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(denso2)
        saida = Dense(1, activation='sigmoid')(dropout2)

        modelo = Model(
            inputs=[entrada_usuario, entrada_item, entrada_conteudo],
            outputs=saida
        )
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return modelo

    def _criar_features_conteudo(self, dados_itens):
        # Combinar título e conteúdo
        textos = dados_itens['Titulo'] + ' ' + dados_itens['Corpo']
        return self.tfidf.fit_transform(textos).toarray()

    def _calcular_matriz_similaridade(self, features_conteudo):
        return cosine_similarity(features_conteudo)

    def treinar(self, dados_treino, dados_itens):
        # Processar features de conteúdo
        features_conteudo = self._criar_features_conteudo(dados_itens)
        self.matriz_similaridade = self._calcular_matriz_similaridade(features_conteudo)

        # Criar interações usuário-item
        for _, linha in dados_treino.iterrows():
            id_usuario = linha['idUsuario']
            historico = eval(linha['historico'])
            self.itens_usuario[id_usuario] = set(historico)

        # Armazenar features dos itens
        for idx, linha in dados_itens.iterrows():
            self.features_item[idx] = {
                'vetor_conteudo': features_conteudo[idx],
                'timestamp': pd.to_datetime(linha['DataPublicacao']).timestamp()
            }

        # Treinar modelo neural
        self.modelo = self._construir_modelo_neural(
            len(self.itens_usuario),
            len(self.features_item)
        )
        # Implementação do treinamento aqui

    def prever(self, id_usuario, n_recomendacoes=10):
        if id_usuario not in self.itens_usuario:
            return self._recomendacoes_usuario_novo()

        # Scores de filtragem colaborativa
        scores_cf = self._obter_scores_colaborativos(id_usuario)
        
        # Scores baseados em conteúdo
        scores_cb = self._obter_scores_conteudo(id_usuario)
        
        # Scores temporais
        scores_temp = self._obter_scores_temporais()

        # Combinar scores
        scores_finais = {}
        for id_item in self.features_item:
            if id_item not in self.itens_usuario[id_usuario]:
                score = (
                    0.4 * scores_cf.get(id_item, 0) +
                    0.4 * scores_cb.get(id_item, 0) +
                    0.2 * scores_temp.get(id_item, 0)
                )
                scores_finais[id_item] = score

        # Ordenar e retornar top N recomendações
        itens_ordenados = sorted(scores_finais.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in itens_ordenados[:n_recomendacoes]]

    def _recomendacoes_usuario_novo(self):
        # Retornar itens mais recentes e populares
        itens_ordenados = sorted(
            self.features_item.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        return [item[0] for item in itens_ordenados[:10]]

    def salvar_modelo(self, caminho):
        with open(caminho, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'itens_usuario': self.itens_usuario,
                'features_item': self.features_item,
                'matriz_similaridade': self.matriz_similaridade
            }, f)

    @classmethod
    def carregar_modelo(cls, caminho):
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
            instancia = cls()
            instancia.modelo = dados['modelo']
            instancia.tfidf = dados['tfidf']
            instancia.itens_usuario = dados['itens_usuario']
            instancia.features_item = dados['features_item']
            instancia.matriz_similaridade = dados['matriz_similaridade']
            return instancia
EOL

cat > projeto/src/modelo/preprocessamento.py << 'EOL'
import pandas as pd
import numpy as np
from datetime import datetime

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None

    def processar_dados_treino(self, arquivos_treino, arquivos_itens):
        # Carregar dados de treino
        dfs_treino = []
        for arquivo in arquivos_treino:
            df = pd.read_csv(arquivo)
            dfs_treino.append(df)
        dados_treino = pd.concat(dfs_treino, ignore_index=True)
        
        # Carregar dados dos itens
        dfs_itens = []
        for arquivo in arquivos_itens:
            df = pd.read_csv(arquivo)
            dfs_itens.append(df)
        dados_itens = pd.concat(dfs_itens, ignore_index=True)
        
        # Processar histórico
        dados_treino['historico'] = dados_treino['historico'].apply(eval)
        dados_treino['historicoTimestamp'] = dados_treino['historicoTimestamp'].apply(eval)
        
        # Processar timestamps
        dados_itens['DataPublicacao'] = pd.to_datetime(dados_itens['DataPublicacao'])
        
        return dados_treino, dados_itens

    def preparar_features_texto(self, dados_itens):
        # Combinar título e conteúdo
        dados_itens['conteudo_texto'] = dados_itens['Titulo'] + ' ' + dados_itens['Corpo']
        return dados_itens
EOL

cat > projeto/src/api/app.py << 'EOL'
from flask import Flask, request, jsonify
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json

app = Flask(__name__)

# Carregar modelo e dados
modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
dados_itens = pd.read_csv('dados/brutos/itens/itens-parte1.csv')

@app.route('/saude', methods=['GET'])
def verificar_saude():
    return jsonify({"status": "saudavel"}), 200

@app.route('/prever', methods=['POST'])
@tratar_excecoes
@validar_entrada_json(['id_usuario'])
def obter_recomendacoes():
    dados = request.get_json()
    id_usuario = dados['id_usuario']
    n_recomendacoes = dados.get('n_recomendacoes', 10)
    
    recomendacoes = modelo.prever(id_usuario, n_recomendacoes)
    
    return jsonify({
        "recomendacoes": recomendacoes
    }), 200

@app.errorhandler(404)
def nao_encontrado(erro):
    return jsonify({"erro": "Rota não encontrada"}), 404

@app.errorhandler(500)
def erro_interno(erro):
    return jsonify({"erro": "Erro interno do servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOL

cat > projeto/src/utils/helpers.py << 'EOL'
import os
import glob
import pandas as pd
from typing import List
from functools import wraps
from flask import jsonify, request

def carregar_arquivos_dados(padrao: str) -> List[str]:
    return glob.glob(padrao)

def combinar_dataframes(arquivos: List[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

def criar_diretorio_se_nao_existe(diretorio: str):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

def validar_entrada_json(campos_obrigatorios):
    def decorador(f):
        @wraps(f)
        def funcao_decorada(*args, **kwargs):
            dados = request.get_json()
            if not dados:
                return jsonify({"erro": "Dados JSON não fornecidos"}), 400
                
            campos_faltantes = [campo for campo in campos_obrigatorios if campo not in dados]
            if campos_faltantes:
                return jsonify({
                    "erro": f"Campos obrigatórios faltando: {', '.join(campos_faltantes)}"
                }), 400
                
            return f(*args, **kwargs)
        return funcao_decorada
    return decorador

def tratar_excecoes(f):
    @wraps(f)
    def funcao_decorada(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"erro": str(e)}), 500
    return funcao_decorada
EOL

cat > projeto/treinar.py << 'EOL'
import os
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento import PreProcessadorDados
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe

def treinar_modelo():
    # Configurar diretórios
    criar_diretorio_se_nao_existe('modelos/modelos_salvos')
    
    # Carregar dados
    arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte_*.csv')
    arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
    
    # Pré-processar dados
    preprocessador = PreProcessadorDados()
    dados_treino, dados_itens = preprocessador.processar_dados_treino(arquivos_treino, arquivos_itens)
    dados_itens = preprocessador.preparar_features_texto(dados_itens)
    
    # Criar e treinar modelo
    modelo = RecomendadorHibrido()
    modelo.treinar(dados_treino, dados_itens)
    
    # Salvar modelo
    modelo.salvar_modelo('modelos/modelos_salvos/recomendador_hibrido')

if __name__ == "__main__":
    treinar_modelo()
EOL

cat > projeto/executar.py << 'EOL'
from src.api.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOL

cat > projeto/testes/teste_modelo.py << 'EOL'
import unittest
import pandas as pd
import numpy as np
from src.modelo.recomendador import RecomendadorHibrido

class TesteRecomendador(unittest.TestCase):
    def setUp(self):
        self.modelo = RecomendadorHibrido()
        
        # Criar dados de exemplo
        self.dados_treino = pd.DataFrame({
            'idUsuario': ['usuario1', 'usuario2'],
            'historico': [['item1', 'item2'], ['item3', 'item4']]
        })
        
        self.dados_itens = pd.DataFrame({
            'Pagina': ['item1', 'item2', 'item3', 'item4'],
            'Titulo': ['Título 1', 'Título 2', 'Título 3', 'Título 4'],
            'Corpo': ['Corpo 1', 'Corpo 2', 'Corpo 3', 'Corpo 4'],
            'DataPublicacao': ['2023-01-01'] * 4
        })

    def teste_inicializacao_modelo(self):
        self.assertIsNotNone(self.modelo)

    def teste_treinamento_modelo(self):
        self.modelo.treinar(self.dados_treino, self.dados_itens)
        self.assertIsNotNone(self.modelo.matriz_similaridade)

    def teste_previsao_modelo(self):
        self.modelo.treinar(self.dados_treino, self.dados_itens)
        previsoes = self.modelo.prever('usuario1', n_recomendacoes=2)
        self.assertEqual(len(previsoes), 2)

if __name__ == '__main__':
    unittest.main()
EOL

cat > projeto/testes/teste_api.py << 'EOL'
import unittest
import json
from src.api.app import app

class TesteAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def teste_verificar_saude(self):
        resposta = self.app.get('/saude')
        dados = json.loads(resposta.data.decode())
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(dados['status'], 'saudavel')

    def teste_endpoint_previsao(self):
        entrada_teste = {
            "id_usuario": "usuario_teste",
            "n_recomendacoes": 5
        }
        
        resposta = self.app.post(
            '/prever',
            data=json.dumps(entrada_teste),
            content_type='application/json'
        )
        
        dados = json.loads(resposta.data.decode())
        self.assertEqual(resposta.status_code, 200)
        self.assertIn('recomendacoes', dados)

if __name__ == '__main__':
    unittest.main()
EOL

cat > projeto/Dockerfile << 'EOL'
FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=producao

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
EOL

cat > projeto/requirements.txt << 'EOL'
tensorflow>=2.5.0
flask>=2.0.1
pandas>=1.3.0
numpy>=1.19.5
pytest>=6.2.5
scikit-learn>=0.24.2
python-dotenv>=0.19.0
EOL

cat > projeto/README.md << 'EOL'
# Sistema de Recomendação Híbrido de Notícias

Sistema de recomendação que combina abordagens de filtragem colaborativa, baseada em conteúdo e temporal.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

1. Treinar o modelo:
```bash
python treinar.py
```

2. Executar a API:
```bash
python executar.py
```

3. Docker:
```bash
docker build -t recomendador-noticias .
docker run -p 8000:8000 recomendador-noticias
```

## Testes

```bash
python -m pytest testes/
```

## Endpoints da API

### Verificação de Saúde
GET /saude

### Previsões
POST /prever
```json
{
    "id_usuario": "string",
    "n_recomendacoes": "integer"
}
```
EOL

# Criar arquivos __init__.py vazios
touch projeto/src/__init__.py
touch projeto/src/modelo/__init__.py
touch projeto/src/api/__init__.py
touch projeto/src/utils/__init__.py
touch projeto/testes/__init__.py

# Tornar o script executável
chmod +x criar_projeto.sh

echo -e "${VERDE}Estrutura do projeto criada com sucesso!${SEM_COR}"
