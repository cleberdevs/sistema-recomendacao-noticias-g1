import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
import pickle
from datetime import datetime
from src.config.mlflow_config import MLflowConfig

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100):
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        self.tfidf = TfidfVectorizer(max_features=dim_features_texto)
        self.itens_usuario = {}
        self.features_item = {}
        self.matriz_similaridade = None
        self.mlflow_config = MLflowConfig()

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        entrada_usuario = Input(shape=(1,))
        entrada_item = Input(shape=(1,))
        entrada_conteudo = Input(shape=(self.dim_features_texto,))

        embedding_usuario = Embedding(n_usuarios, self.dim_embedding)(entrada_usuario)
        embedding_item = Embedding(n_itens, self.dim_embedding)(entrada_item)

        usuario_flat = Flatten()(embedding_usuario)
        item_flat = Flatten()(embedding_item)

        concat = Concatenate()([usuario_flat, item_flat, entrada_conteudo])

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

    def treinar(self, dados_treino, dados_itens):
        self.mlflow_config.setup_mlflow()
        
        with self.mlflow_config.iniciar_run(run_name="treinamento_modelo"):
            # Processar features de conteúdo
            features_conteudo = self._criar_features_conteudo(dados_itens)
            self.matriz_similaridade = self._calcular_matriz_similaridade(features_conteudo)

            # Registrar parâmetros
            self.mlflow_config.log_parametros({
                "dim_embedding": self.dim_embedding,
                "dim_features_texto": self.dim_features_texto,
                "tamanho_dados_treino": len(dados_treino)
            })

            # Criar e treinar modelo neural
            self.modelo = self._construir_modelo_neural(
                len(set(dados_treino['idUsuario'])),
                len(dados_itens)
            )
            
            # Treinar modelo (implementar lógica de treinamento aqui)
            # historia_treino = self.modelo.fit(...)
            
            # Log de métricas
            metricas = {
                "loss": 0.0,  # Substituir pelos valores reais
                "accuracy": 0.0  # Substituir pelos valores reais
            }
            self.mlflow_config.log_metricas(metricas)

    def prever(self, id_usuario, n_recomendacoes=10):
        if id_usuario not in self.itens_usuario:
            return self._recomendacoes_usuario_novo()

        # Implementar lógica de previsão
        return []

    def salvar_modelo(self, caminho):
        with open(caminho, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'itens_usuario': self.itens_usuario,
                'features_item': self.features_item,
                'matriz_similaridade': self.matriz_similaridade
            }, f)
        self.mlflow_config.log_artefato(caminho)

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

