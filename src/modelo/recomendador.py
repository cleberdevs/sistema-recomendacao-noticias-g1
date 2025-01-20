import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from datetime import datetime
import mlflow
from src.config.mlflow_config import MLflowConfig
import logging

logger = logging.getLogger(__name__)

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        self.tfidf = TfidfVectorizer(
            max_features=dim_features_texto,
            stop_words='portuguese'
        )
        self.itens_usuario = {}
        self.features_item = {}
        self.matriz_similaridade = None
        self.mlflow_config = mlflow_config if mlflow_config else MLflowConfig()
        logger.info("Inicializando RecomendadorHibrido")

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói a arquitetura do modelo neural.
        
        Args:
            n_usuarios: Número de usuários únicos
            n_itens: Número de itens únicos
            
        Returns:
            Modelo Keras compilado
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        try:
            # Input layers
            entrada_usuario = Input(shape=(1,))
            entrada_item = Input(shape=(1,))
            entrada_conteudo = Input(shape=(self.dim_features_texto,))

            # Embedding layers
            embedding_usuario = Embedding(
                n_usuarios, 
                self.dim_embedding,
                name='embedding_usuario'
            )(entrada_usuario)
            
            embedding_item = Embedding(
                n_itens, 
                self.dim_embedding,
                name='embedding_item'
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten()(embedding_usuario)
            item_flat = Flatten()(embedding_item)

            # Concatenate
            concat = Concatenate()([usuario_flat, item_flat, entrada_conteudo])

            # Dense layers
            denso1 = Dense(128, activation='relu', name='dense_1')(concat)
            dropout1 = Dropout(0.3)(denso1)
            denso2 = Dense(64, activation='relu', name='dense_2')(dropout1)
            dropout2 = Dropout(0.2)(denso2)
            saida = Dense(1, activation='sigmoid', name='output')(dropout2)

            # Create model
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida
            )
            
            # Compile
            modelo.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )
            
            logger.info("Modelo neural construído com sucesso")
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise

    def _criar_features_conteudo(self, dados_itens):
        """
        Cria features de conteúdo usando TF-IDF.
        
        Args:
            dados_itens: DataFrame Spark com os dados dos itens
            
        Returns:
            array: Matrix de features TF-IDF
        """
        logger.info("Criando features de conteúdo")
        try:
            # Converter DataFrame Spark para Pandas
            df_pandas = dados_itens.select("conteudo_texto").toPandas()
            
            # Preencher valores nulos
            textos = df_pandas['conteudo_texto'].fillna('')
            
            # Criar features TF-IDF
            features = self.tfidf.fit_transform(textos).toarray()
            
            logger.info(f"Features de conteúdo criadas com forma: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Erro ao criar features de conteúdo: {str(e)}")
            raise

    def _calcular_matriz_similaridade(self, features_conteudo):
        """
        Calcula matriz de similaridade entre itens.
        
        Args:
            features_conteudo: Matrix de features TF-IDF
            
        Returns:
            array: Matriz de similaridade
        """
        logger.info("Calculando matriz de similaridade")
        return cosine_similarity(features_conteudo)

    def _configurar_callbacks(self):
        """
        Configura callbacks para o treinamento.
        
        Returns:
            list: Lista de callbacks Keras
        """
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.00001
            )
        ]

    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo híbrido.
        
        Args:
            dados_treino: DataFrame Spark com dados de treino
            dados_itens: DataFrame Spark com dados dos itens
            
        Returns:
            História do treinamento
        """
        logger.info("Iniciando treinamento do modelo")
        
        try:
            # Verificar se os DataFrames são válidos
            if dados_treino is None or dados_itens is None:
                raise ValueError("Dados de treino ou itens são None")
                
            # Converter dados de treino para Pandas se necessário
            if hasattr(dados_treino, 'toPandas'):
                dados_treino = dados_treino.toPandas()
            
            # Processar features de conteúdo
            features_conteudo = self._criar_features_conteudo(dados_itens)
            self.matriz_similaridade = self._calcular_matriz_similaridade(features_conteudo)

            # Criar mapeamento usuário-item
            for _, linha in dados_treino.iterrows():
                self.itens_usuario[linha['idUsuario']] = set(linha['historico'])

            # Armazenar features dos itens
            dados_itens_pandas = dados_itens.toPandas() if hasattr(dados_itens, 'toPandas') else dados_itens
            for idx, linha in dados_itens_pandas.iterrows():
                self.features_item[idx] = {
                    'vetor_conteudo': features_conteudo[idx],
                    'timestamp': linha['DataPublicacao'].timestamp()
                }

            # Construir e treinar modelo neural
            n_usuarios = len(self.itens_usuario)
            n_itens = len(self.features_item)
            
            self.modelo = self._construir_modelo_neural(n_usuarios, n_itens)
            
            # Preparar dados de treino
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino(
                dados_treino,
                features_conteudo
            )

            # Configurar callbacks
            callbacks = self._configurar_callbacks()

            # Treinar modelo
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                epochs=5,
                batch_size=64,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Treinamento concluído com sucesso")
            return historia

        except Exception as e:
            logger.error(f"Erro durante o treinamento: {str(e)}")
            raise

    def _preparar_dados_treino(self, dados_treino, features_conteudo):
        """
        Prepara os dados para treinamento do modelo neural.
        
        Args:
            dados_treino: DataFrame com dados de treino
            features_conteudo: Matrix de features TF-IDF
            
        Returns:
            tuple: Arrays com dados de treino preparados
        """
        X_usuario, X_item, X_conteudo, y = [], [], [], []
        
        for _, linha in dados_treino.iterrows():
            usuario_id = linha['idUsuario']
            historico = linha['historico']
            
            # Amostras positivas
            for item_id in historico:
                X_usuario.append(usuario_id)
                X_item.append(item_id)
                X_conteudo.append(features_conteudo[item_id])
                y.append(1)
            
            # Amostras negativas balanceadas
            negativos = self._gerar_amostras_negativas(historico, len(features_conteudo))
            for item_id in negativos:
                X_usuario.append(usuario_id)
                X_item.append(item_id)
                X_conteudo.append(features_conteudo[item_id])
                y.append(0)
        
        return (
            np.array(X_usuario),
            np.array(X_item),
            np.array(X_conteudo),
            np.array(y)
        )

    def _gerar_amostras_negativas(self, historico_positivo, n_itens, n_amostras=None):
        """
        Gera amostras negativas aleatórias.
        
        Args:
            historico_positivo: Lista de itens já vistos
            n_itens: Número total de itens
            n_amostras: Número de amostras negativas a gerar
            
        Returns:
            array: Índices das amostras negativas
        """
        if n_amostras is None:
            n_amostras = len(historico_positivo)
            
        todos_itens = set(range(n_itens))
        itens_negativos = list(todos_itens - set(historico_positivo))
        
        return np.random.choice(
            itens_negativos,
            size=min(n_amostras, len(itens_negativos)),
            replace=False
        )

    def prever(self, id_usuario, n_recomendacoes=10):
        """
        Gera recomendações para um usuário.
        
        Args:
            id_usuario: ID do usuário
            n_recomendacoes: Número de recomendações a gerar
            
        Returns:
            list: Lista de IDs dos itens recomendados
        """
        logger.info(f"Gerando previsões para usuário {id_usuario}")
        try:
            if id_usuario not in self.itens_usuario:
                return self._recomendacoes_usuario_novo()

            # Preparar dados para previsão
            todos_itens = list(self.features_item.keys())
            usuario_input = np.array([id_usuario] * len(todos_itens))
            item_input = np.array(todos_itens)
            conteudo_input = np.array([self.features_item[i]['vetor_conteudo'] 
                                     for i in todos_itens])

            # Gerar previsões em lotes
            batch_size = 1024
            previsoes = []
            
            for i in range(0, len(todos_itens), batch_size):
                batch_end = min(i + batch_size, len(todos_itens))
                batch_previsoes = self.modelo.predict(
                    [
                        usuario_input[i:batch_end],
                        item_input[i:batch_end],
                        conteudo_input[i:batch_end]
                    ],
                    verbose=0
                )
                previsoes.extend(batch_previsoes.flatten())

            # Filtrar itens já vistos
            itens_vistos = self.itens_usuario[id_usuario]
            scores = [(item, score) for item, score in zip(todos_itens, previsoes)
                     if item not in itens_vistos]

            # Ordenar e retornar top N
            recomendacoes = sorted(scores, key=lambda x: x[1], reverse=True)
            return [item for item, _ in recomendacoes[:n_recomendacoes]]

        except Exception as e:
            logger.error(f"Erro ao gerar previsões: {str(e)}")
            raise

    def _recomendacoes_usuario_novo(self):
        """
        Gera recomendações para usuários novos.
        
        Returns:
            list: Lista de IDs dos itens recomendados
        """
        logger.info("Gerando recomendações para usuário novo")
        try:
            # Ordenar itens por timestamp (recência)
            itens_ordenados = sorted(
                self.features_item.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )
            return [item[0] for item in itens_ordenados[:10]]
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações para usuário novo: {str(e)}")
            raise

    def salvar_modelo(self, caminho):
        """
        Salva o modelo em disco.
        
        Args:
            caminho: Caminho onde salvar o modelo
        """
        logger.info(f"Salvando modelo em {caminho}")
        try:
            dados_modelo = {
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'itens_usuario': self.itens_usuario,
                'features_item': self.features_item,
                'matriz_similaridade': self.matriz_similaridade,
                'dim_embedding': self.dim_embedding,
                'dim_features_texto': self.dim_features_texto
            }
            
            with open(caminho, 'wb') as f:
                pickle.dump(dados_modelo, f)
            
            if mlflow.active_run():
                self.mlflow_config.log_artefato(caminho)
            logger.info("Modelo salvo com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise

    @classmethod
    def carregar_modelo(cls, caminho):
        """
        Carrega um modelo salvo.
        
        Args:
            caminho: Caminho do modelo salvo
            
        Returns:
            RecomendadorHibrido: Instância do modelo carregado
        """
        logger.info(f"Carregando modelo de {caminho}")
        try:
            with open(caminho, 'rb') as f:
                dados_modelo = pickle.load(f)
                
            instancia = cls(
                dim_embedding=dados_modelo['dim_embedding'],
                dim_features_texto=dados_modelo['dim_features_texto']
            )
            instancia.modelo = dados_modelo['modelo']
            instancia.tfidf = dados_modelo['tfidf']
            instancia.itens_usuario = dados_modelo['itens_usuario']
            instancia.features_item = dados_modelo['features_item']
            instancia.matriz_similaridade = dados_modelo['matriz_similaridade']
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise