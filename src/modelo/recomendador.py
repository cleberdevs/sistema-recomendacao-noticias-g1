'''import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
import pickle
from datetime import datetime
import mlflow
from src.config.mlflow_config import MLflowConfig
import logging

logging.basicConfig(level=logging.INFO)
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

            # Dense layers with better regularization
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
            
            # Compile with better metrics
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
        logger.info("Criando features de conteúdo")
        try:
            textos = dados_itens['conteudo_texto'].fillna('')
            return self.tfidf.fit_transform(textos).toarray()
        except Exception as e:
            logger.error(f"Erro ao criar features de conteúdo: {str(e)}")
            raise

    def _calcular_matriz_similaridade(self, features_conteudo):
        logger.info("Calculando matriz de similaridade")
        return cosine_similarity(features_conteudo)

    def treinar(self, dados_treino, dados_itens):
        logger.info("Iniciando treinamento do modelo")
        try:
            # Processar features de conteúdo
            features_conteudo = self._criar_features_conteudo(dados_itens)
            self.matriz_similaridade = self._calcular_matriz_similaridade(features_conteudo)

            # Registrar parâmetros se MLflow estiver ativo
            if mlflow.active_run():
                parametros = {
                    "dim_embedding": self.dim_embedding,
                    "dim_features_texto": self.dim_features_texto,
                    "tamanho_dados_treino": len(dados_treino),
                    "tamanho_dados_itens": len(dados_itens)
                }
                self.mlflow_config.log_parametros(parametros)

            # Criar mapeamento usuário-item
            for _, linha in dados_treino.iterrows():
                self.itens_usuario[linha['idUsuario']] = set(linha['historico'])

            # Armazenar features dos itens
            for idx, linha in dados_itens.iterrows():
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

            # Callbacks para early stopping e redução de learning rate
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

            # Log métricas se MLflow estiver ativo
            if mlflow.active_run():
                metricas = {
                    "loss_final": historia.history['loss'][-1],
                    "accuracy_final": historia.history['accuracy'][-1],
                    "val_loss_final": historia.history['val_loss'][-1],
                    "val_accuracy_final": historia.history['val_accuracy'][-1],
                    "precision_final": historia.history['precision'][-1],
                    "recall_final": historia.history['recall'][-1]
                }
                self.mlflow_config.log_metricas(metricas)
            
            logger.info("Treinamento concluído com sucesso")
            return historia

        except Exception as e:
            logger.error(f"Erro durante o treinamento: {str(e)}")
            raise

    def _configurar_callbacks(self):
        """Configura callbacks para o treinamento"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
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

    def _preparar_dados_treino(self, dados_treino, features_conteudo):
        """Prepara os dados para treinamento do modelo neural"""
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
        
        return np.array(X_usuario), np.array(X_item), np.array(X_conteudo), np.array(y)

    def _gerar_amostras_negativas(self, historico_positivo, n_itens, n_amostras=None):
        """Gera amostras negativas aleatórias"""
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

            # Gerar previsões em lotes para melhor performance
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
        """Recomendações para usuários novos baseadas em popularidade e recência"""
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
            raise'''

'''import numpy as np
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
import nltk

try:
    nltk.download('stopwords')
except:
    pass
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        
        try:
            stop_words_pt = stopwords.words('portuguese')
        except:
            # Lista básica de stopwords em português caso o download falhe
            stop_words_pt = ['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 
                           'aquilo', 'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 
                           'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 
                           'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 
                           'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 
                           'eu', 'foi', 'foram', 'havia', 'isso', 'isto', 'já', 'lhe', 
                           'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 
                           'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no', 'nos', 
                           'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 
                           'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 
                           'quando', 'que', 'quem', 'são', 'se', 'seja', 'sejam', 'sem', 
                           'ser', 'será', 'seu', 'seus', 'só', 'sua', 'suas', 'também', 
                           'te', 'tem', 'tendo', 'tenha', 'ter', 'teu', 'teus', 'ti', 
                           'tua', 'tuas', 'um', 'uma', 'umas', 'uns', 'você', 'vocês', 
                           'vos', 'à', 'às']
            logger.warning("Falha ao carregar stopwords do NLTK, usando lista padrão")

        self.tfidf = TfidfVectorizer(
            max_features=dim_features_texto,
            stop_words=stop_words_pt
        )
        
        self.itens_usuario = {}
        self.features_item = {}
        self.mlflow_config = mlflow_config if mlflow_config else MLflowConfig()
        logger.info("RecomendadorHibrido inicializado")

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural.
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
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise

    def _criar_features_conteudo_pandas(self, dados_itens_pd):
        """
        Cria features de conteúdo usando dados já convertidos para pandas.
        """
        logger.info("Criando features de conteúdo")
        try:
            textos = dados_itens_pd['conteudo_texto'].fillna('').values
            features = self.tfidf.fit_transform(textos).toarray()
            logger.info(f"Features de conteúdo criadas com forma: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"Erro ao criar features de conteúdo: {str(e)}")
            raise

    def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos de usuários e itens em lotes.
        """
        logger.info("Criando mapeamentos")
        try:
            # Processar usuários em lotes
            batch_size = 1000
            n_usuarios_processados = 0
            n_usuarios_com_historico = 0
            
            for i in range(0, len(dados_treino_pd), batch_size):
                batch = dados_treino_pd.iloc[i:i+batch_size]
                for _, linha in batch.iterrows():
                    try:
                        historico = linha['historico']
                        if historico is not None and isinstance(historico, (list, np.ndarray)):
                            historico_valido = [h for h in historico if isinstance(h, (int, np.integer))]
                            if historico_valido:
                                self.itens_usuario[linha['idUsuario']] = set(historico_valido)
                                n_usuarios_com_historico += 1
                            else:
                                self.itens_usuario[linha['idUsuario']] = set()
                                logger.warning(f"Usuário {linha['idUsuario']} com histórico vazio após validação")
                        else:
                            self.itens_usuario[linha['idUsuario']] = set()
                            logger.warning(f"Usuário {linha['idUsuario']} com histórico inválido")
                        n_usuarios_processados += 1
                    except Exception as e:
                        logger.error(f"Erro ao processar usuário {linha.get('idUsuario', 'unknown')}: {str(e)}")

            logger.info(f"Usuários processados: {n_usuarios_processados}, com histórico: {n_usuarios_com_historico}")

            # Processar itens em lotes
            n_itens_processados = 0
            for i in range(0, len(dados_itens_pd), batch_size):
                batch = dados_itens_pd.iloc[i:i+batch_size]
                for idx, linha in batch.iterrows():
                    try:
                        timestamp = linha['DataPublicacao'].timestamp() if pd.notnull(linha['DataPublicacao']) else 0
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': timestamp
                        }
                        n_itens_processados += 1
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Erro ao processar item {idx}: {str(e)}")
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': 0
                        }

            logger.info(f"Mapeamentos finalizados: {len(self.itens_usuario)} usuários, {n_itens_processados} itens processados")
            
            if not self.itens_usuario:
                raise ValueError("Nenhum usuário com histórico válido encontrado")
                
        except Exception as e:
            logger.error(f"Erro ao criar mapeamentos: {str(e)}")
            raise

    def _configurar_callbacks(self):
        """Configura callbacks para treinamento."""
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
        Treina o modelo.
        """
        logger.info("Iniciando treinamento")
        try:
            # Coletar dados do Spark primeiro
            logger.info("Coletando dados do Spark")
            try:
                # Debug: mostrar schema e primeiros registros
                logger.info(f"Schema dados_treino: {dados_treino.schema}")
                logger.info(f"Schema dados_itens: {dados_itens.schema}")
                
                # Verificar dados antes da coleta
                dados_treino.cache()
                dados_itens.cache()
                
                logger.info(f"Número de registros em dados_treino: {dados_treino.count()}")
                logger.info(f"Número de registros em dados_itens: {dados_itens.count()}")
                
                # Mostrar alguns exemplos
                logger.info("Exemplos de dados_treino:")
                for row in dados_treino.limit(5).collect():
                    logger.info(f"Registro: {row}")

                # Coletar dados com validação
                dados_treino_list = dados_treino.select("idUsuario", "historico").collect()
                dados_itens_list = dados_itens.select("conteudo_texto", "DataPublicacao").collect()
                
                # Converter para formato mais conveniente e filtrar registros inválidos
                dados_treino_validos = []
                for row in dados_treino_list:
                    historico = row['historico']
                    if historico is not None and isinstance(historico, (list, np.ndarray)) and len(historico) > 0:
                        dados_treino_validos.append({
                            'idUsuario': row['idUsuario'],
                            'historico': historico
                        })
                    else:
                        logger.warning(f"Registro inválido encontrado: {row}")
                
                if not dados_treino_validos:
                    raise ValueError("Nenhum dado de treino válido encontrado")
                    
                dados_treino_pd = pd.DataFrame(dados_treino_validos)
                
                dados_itens_pd = pd.DataFrame([
                    {
                        'conteudo_texto': row['conteudo_texto'] if row['conteudo_texto'] is not None else '',
                        'DataPublicacao': row['DataPublicacao']
                    }
                    for row in dados_itens_list
                ])
                
                logger.info(f"Dados coletados: {len(dados_treino_pd)} registros de treino válidos, {len(dados_itens_pd)} itens")
                
            except Exception as e:
                logger.error(f"Erro ao coletar dados do Spark: {str(e)}")
                raise

            # Processar features de conteúdo
            logger.info("Processando features de texto")
            features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)

            # Criar mapeamentos em lotes
            logger.info("Criando mapeamentos usuário-item")
            self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)

            # Construir modelo
            n_usuarios = len(self.itens_usuario)
            n_itens = len(self.features_item)
            self.modelo = self._construir_modelo_neural(n_usuarios, n_itens)

            # Preparar dados em lotes
            logger.info("Preparando dados de treino")
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd,
                features_conteudo
            )

            # Configurar callbacks
            callbacks = self._configurar_callbacks()

            # Treinar modelo
            logger.info("Iniciando treinamento do modelo neural")
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
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise

    def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, batch_size=1000):
        """
        Prepara dados de treino em lotes para evitar sobrecarga de memória.
        """
        X_usuario, X_item, X_conteudo, y = [], [], [], []
        
        for i in range(0, len(dados_treino_pd), batch_size):
            batch = dados_treino_pd.iloc[i:i+batch_size]
            
            for _, linha in batch.iterrows():
                usuario_id = linha['idUsuario']
                historico = linha['historico']
                
                if historico is None or len(historico) == 0:
                    continue
                
                # Amostras positivas
                for item_id in historico:
                    if item_id >= len(features_conteudo):
                        continue
                    X_usuario.append(usuario_id)
                    X_item.append(item_id)
                    X_conteudo.append(features_conteudo[item_id])
                    y.append(1)
                
                # Amostras negativas
                negativos = self._gerar_amostras_negativas(historico, len(features_conteudo))
                for item_id in negativos:
                    X_usuario.append(usuario_id)
                    X_item.append(item_id)
                    X_conteudo.append(features_conteudo[item_id])
                    y.append(0)
        
        if len(X_usuario) == 0:
            raise ValueError("Nenhum dado de treino válido encontrado")
        
        return (np.array(X_usuario), np.array(X_item), 
                np.array(X_conteudo), np.array(y))

    def _gerar_amostras_negativas(self, historico_positivo, n_itens, n_amostras=None):
        """
        Gera amostras negativas para treinamento.
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
        """
        logger.info(f"Gerando previsões para usuário {id_usuario}")
        try:
            if id_usuario not in self.itens_usuario:
                return self._recomendacoes_usuario_novo()

            # Preparar dados
            todos_itens = list(self.features_item.keys())
            itens_nao_vistos = [i for i in todos_itens 
                               if i not in self.itens_usuario[id_usuario]]
            
            # Gerar previsões em lotes
            batch_size = 1024
            previsoes = []
            
            for i in range(0, len(itens_nao_vistos), batch_size):
                fim = min(i + batch_size, len(itens_nao_vistos))
                batch_itens = itens_nao_vistos[i:fim]
                
                usuario_input = np.array([id_usuario] * len(batch_itens))
                item_input = np.array(batch_itens)
                conteudo_input = np.array([
                    self.features_item[j]['vetor_conteudo'] 
                    for j in batch_itens
                ])

                batch_previsoes = self.modelo.predict(
                    [usuario_input, item_input, conteudo_input],
                    verbose=0
                )
                previsoes.extend(batch_previsoes.flatten())

            # Combinar scores e retornar top N
            scores = list(zip(itens_nao_vistos, previsoes))
            recomendacoes = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [item for item, _ in recomendacoes[:n_recomendacoes]]

        except Exception as e:
            logger.error(f"Erro ao gerar previsões: {str(e)}")
            raise

    def _recomendacoes_usuario_novo(self):
        """
        Recomendações para usuários novos.
        """
        logger.info("Gerando recomendações para usuário novo")
        try:
            # Ordenar por recência
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
        """
        logger.info(f"Salvando modelo em {caminho}")
        try:
            dados_modelo = {
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'itens_usuario': self.itens_usuario,
                'features_item': self.features_item,
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
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise'''

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
import nltk
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    nltk.download('stopwords')
except:
    pass
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        
        try:
            stop_words_pt = stopwords.words('portuguese')
        except:
            # Lista básica de stopwords em português caso o download falhe
            stop_words_pt = ['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 
                           'aquilo', 'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 
                           'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 
                           'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 
                           'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 
                           'eu', 'foi', 'foram', 'havia', 'isso', 'isto', 'já', 'lhe', 
                           'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 
                           'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no', 'nos', 
                           'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 
                           'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 
                           'quando', 'que', 'quem', 'são', 'se', 'seja', 'sejam', 'sem', 
                           'ser', 'será', 'seu', 'seus', 'só', 'sua', 'suas', 'também', 
                           'te', 'tem', 'tendo', 'tenha', 'ter', 'teu', 'teus', 'ti', 
                           'tua', 'tuas', 'um', 'uma', 'umas', 'uns', 'você', 'vocês', 
                           'vos', 'à', 'às']
            logger.warning("Falha ao carregar stopwords do NLTK, usando lista padrão")

        self.tfidf = TfidfVectorizer(
            max_features=dim_features_texto,
            stop_words=stop_words_pt
        )
        
        self.itens_usuario = {}
        self.features_item = {}
        self.mlflow_config = mlflow_config if mlflow_config else MLflowConfig()
        logger.info("RecomendadorHibrido inicializado")

    def _tentar_reconectar_spark(self, spark, max_tentativas=3):
        """
        Tenta reconectar ao Spark em caso de falha.
        """
        logger.info("Tentando reconectar ao Spark")
        for tentativa in range(max_tentativas):
            try:
                # Tentar criar nova sessão
                if spark._jsc is None or spark._jsc._sc is None:
                    spark = SparkSession.builder \
                        .appName("RecomendadorNoticias") \
                        .config("spark.driver.memory", "4g") \
                        .config("spark.executor.memory", "4g") \
                        .getOrCreate()
                
                # Testar conexão
                spark.sql("SELECT 1").collect()
                logger.info("Reconexão com Spark bem-sucedida")
                return spark
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1} de reconexão falhou: {str(e)}")
                time.sleep(5)  # Esperar um pouco antes de tentar novamente
        
        raise ConnectionError("Não foi possível reconectar ao Spark")

    def _coletar_dados_spark_seguro(self, dados_spark, colunas, batch_size=1000):
        """
        Coleta dados do Spark de forma segura, em lotes.
        """
        logger.info(f"Coletando dados em lotes: {colunas}")
        resultados = []
        
        try:
            total_registros = dados_spark.count()
            logger.info(f"Total de registros a coletar: {total_registros}")
            
            for offset in range(0, total_registros, batch_size):
                try:
                    batch = dados_spark.select(*colunas) \
                        .limit(batch_size).offset(offset) \
                        .collect()
                    resultados.extend(batch)
                    logger.info(f"Coletados {len(resultados)} de {total_registros} registros")
                except Exception as e:
                    logger.warning(f"Erro ao coletar lote {offset}: {str(e)}")
                    time.sleep(2)  # Pequena pausa antes de tentar o próximo lote
            
            return resultados
        except Exception as e:
            logger.error(f"Erro na coleta de dados: {str(e)}")
            raise

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural.
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
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise

    def _criar_features_conteudo_pandas(self, dados_itens_pd):
        """
        Cria features de conteúdo usando dados já convertidos para pandas.
        """
        logger.info("Criando features de conteúdo")
        try:
            textos = dados_itens_pd['conteudo_texto'].fillna('').values
            features = self.tfidf.fit_transform(textos).toarray()
            logger.info(f"Features de conteúdo criadas com forma: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"Erro ao criar features de conteúdo: {str(e)}")
            raise

    '''def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos de usuários e itens em lotes.
        """
        logger.info("Criando mapeamentos")
        try:
            # Processar usuários em lotes
            batch_size = 1000
            n_usuarios_processados = 0
            n_usuarios_com_historico = 0
            
            for i in range(0, len(dados_treino_pd), batch_size):
                batch = dados_treino_pd.iloc[i:i+batch_size]
                for _, linha in batch.iterrows():
                    try:
                        historico = linha['historico']
                        if historico is not None and isinstance(historico, (list, np.ndarray)):
                            historico_valido = [h for h in historico if isinstance(h, (int, np.integer))]
                            if historico_valido:
                                self.itens_usuario[linha['idUsuario']] = set(historico_valido)
                                n_usuarios_com_historico += 1
                            else:
                                self.itens_usuario[linha['idUsuario']] = set()
                                logger.warning(f"Usuário {linha['idUsuario']} com histórico vazio após validação")
                        else:
                            self.itens_usuario[linha['idUsuario']] = set()
                            logger.warning(f"Usuário {linha['idUsuario']} com histórico inválido")
                        n_usuarios_processados += 1
                    except Exception as e:
                        logger.error(f"Erro ao processar usuário {linha.get('idUsuario', 'unknown')}: {str(e)}")

            logger.info(f"Usuários processados: {n_usuarios_processados}, com histórico: {n_usuarios_com_historico}")

            # Processar itens em lotes
            n_itens_processados = 0
            for i in range(0, len(dados_itens_pd), batch_size):
                batch = dados_itens_pd.iloc[i:i+batch_size]
                for idx, linha in batch.iterrows():
                    try:
                        timestamp = linha['DataPublicacao'].timestamp() if pd.notnull(linha['DataPublicacao']) else 0
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': timestamp
                        }
                        n_itens_processados += 1
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Erro ao processar item {idx}: {str(e)}")
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': 0
                        }

            logger.info(f"Mapeamentos finalizados: {len(self.itens_usuario)} usuários, {n_itens_processados} itens processados")
            
            if not self.itens_usuario:
                raise ValueError("Nenhum usuário com histórico válido encontrado")
                
        except Exception as e:
            logger.error(f"Erro ao criar mapeamentos: {str(e)}")
            raise'''
    def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos de usuários e itens em lotes.
        """
        logger.info("Criando mapeamentos")
        try:
            # Processar usuários em lotes
            batch_size = 1000
            n_usuarios_processados = 0
            n_usuarios_com_historico = 0
            historicos_vazios = 0
            
            for i in range(0, len(dados_treino_pd), batch_size):
                batch = dados_treino_pd.iloc[i:i+batch_size]
                for _, linha in batch.iterrows():
                    try:
                        historico = linha['historico']
                        if historico is not None and isinstance(historico, (list, np.ndarray)):
                            historico_valido = [h for h in historico if isinstance(h, (int, np.integer))]
                            if historico_valido:
                                self.itens_usuario[linha['idUsuario']] = set(historico_valido)
                                n_usuarios_com_historico += 1
                            else:
                                historicos_vazios += 1
                                self.itens_usuario[linha['idUsuario']] = set()
                        else:
                            historicos_vazios += 1
                            self.itens_usuario[linha['idUsuario']] = set()
                        n_usuarios_processados += 1
                    except Exception as e:
                        logger.error(f"Erro ao processar usuário {linha.get('idUsuario', 'unknown')}: {str(e)}")

            logger.info(f"Estatísticas de processamento:")
            logger.info(f"- Usuários processados: {n_usuarios_processados}")
            logger.info(f"- Usuários com histórico: {n_usuarios_com_historico}")
            logger.info(f"- Históricos vazios: {historicos_vazios}")

            if n_usuarios_com_historico == 0:
                raise ValueError("Nenhum usuário com histórico válido encontrado após processamento")

            # Processar itens em lotes
            n_itens_processados = 0
            for i in range(0, len(dados_itens_pd), batch_size):
                batch = dados_itens_pd.iloc[i:i+batch_size]
                for idx, linha in batch.iterrows():
                    try:
                        timestamp = linha['DataPublicacao'].timestamp() if pd.notnull(linha['DataPublicacao']) else 0
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': timestamp
                        }
                        n_itens_processados += 1
                    except Exception as e:
                        logger.warning(f"Erro ao processar item {idx}: {str(e)}")
                        self.features_item[idx] = {
                            'vetor_conteudo': features_conteudo[idx],
                            'timestamp': 0
                        }

            logger.info(f"Items processados: {n_itens_processados}")
            
        except Exception as e:
            logger.error(f"Erro ao criar mapeamentos: {str(e)}")
            raise

    def _configurar_callbacks(self):
        """Configura callbacks para treinamento."""
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

    '''def treinar(self, dados_treino, dados_itens):
        
        """
        Treina o modelo com melhor gestão de recursos e reconexão automática.
        """
        logger.info("Iniciando treinamento")
        try:
            # Coletar dados do Spark de forma segura
            logger.info("Coletando dados do Spark")
            try:
                # Configurar Spark para melhor estabilidade
                spark = dados_treino.sparkSession
                
                # Cache dos DataFrames
                dados_treino.cache()
                dados_itens.cache()
                
                # Coletar dados em lotes
                dados_treino_list = self._coletar_dados_spark_seguro(
                    dados_treino, 
                    ["idUsuario", "historico"],
                    batch_size=1000
                )
                dados_itens_list = self._coletar_dados_spark_seguro(
                    dados_itens,
                    ["conteudo_texto", "DataPublicacao"],
                    batch_size=1000
                )
                
                # Converter para formato mais conveniente e filtrar registros inválidos
                dados_treino_validos = []
                for row in dados_treino_list:
                    historico = row['historico']
                    if historico is not None and isinstance(historico, (list, np.ndarray)) and len(historico) > 0:
                        dados_treino_validos.append({
                            'idUsuario': row['idUsuario'],
                            'historico': historico
                        })
                    else:
                        logger.warning(f"Registro inválido encontrado: {row}")
                
                if not dados_treino_validos:
                    raise ValueError("Nenhum dado de treino válido encontrado")
                    
                dados_treino_pd = pd.DataFrame(dados_treino_validos)
                
                dados_itens_pd = pd.DataFrame([
                    {
                        'conteudo_texto': row['conteudo_texto'] if row['conteudo_texto'] is not None else '',
                        'DataPublicacao': row['DataPublicacao']
                    }
                    for row in dados_itens_list
                ])
                
                logger.info(f"Dados coletados: {len(dados_treino_pd)} registros de treino válidos, {len(dados_itens_pd)} itens")
                
            except Exception as e:
                logger.error(f"Erro ao coletar dados do Spark: {str(e)}")
                raise
            finally:
                # Limpar cache
                try:
                    dados_treino.unpersist()
                    dados_itens.unpersist()
                except:
                    pass

            # Processar features de conteúdo
            logger.info("Processando features de texto")
            features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)

            # Criar mapeamentos em lotes
            logger.info("Criando mapeamentos usuário-item")
            self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)

            # Construir modelo
            n_usuarios = len(self.itens_usuario)
            n_itens = len(self.features_item)
            self.modelo = self._construir_modelo_neural(n_usuarios, n_itens)

            # Preparar dados em lotes
            logger.info("Preparando dados de treino")
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd,
                features_conteudo
            )

            # Configurar callbacks
            callbacks = self._configurar_callbacks()

            # Treinar modelo
            logger.info("Iniciando treinamento do modelo neural")
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
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise'''
    
    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo com melhor gestão de recursos e reconexão automática.
        """
        logger.info("Iniciando treinamento")
        try:
            # Coletar dados do Spark de forma segura
            logger.info("Coletando dados do Spark")
            try:
                # Validação prévia dos dados
                n_registros_total = dados_treino.count()
                
                # Garantir que a coluna historico seja um array
                dados_treino = dados_treino.withColumn(
                    "historico",
                    F.when(F.col("historico").isNull(), F.array()).otherwise(F.col("historico"))
                )
                
                # Contar registros válidos
                n_registros_validos = dados_treino.filter(
                    F.size(F.col('historico')) > 0
                ).count()
                
                logger.info(f"Total de registros: {n_registros_total}")
                logger.info(f"Registros válidos: {n_registros_validos}")
                
                if n_registros_validos == 0:
                    raise ValueError("Nenhum registro com histórico válido encontrado nos dados")
                
                # Filtrar apenas registros válidos antes de coletar
                dados_treino_filtrado = dados_treino.filter(
                    F.size(F.col('historico')) > 0
                )
                
                # Coletar dados em lotes
                dados_treino_list = []
                batch_size = 1000
                
                # Usar coalesce para otimizar a coleta
                dados_treino_filtrado = dados_treino_filtrado.coalesce(1)
                
                for offset in range(0, n_registros_validos, batch_size):
                    batch = dados_treino_filtrado.select("idUsuario", "historico") \
                        .limit(batch_size).offset(offset)
                    dados_treino_list.extend(batch.collect())
                
                dados_itens_list = self._coletar_dados_spark_seguro(
                    dados_itens,
                    ["conteudo_texto", "DataPublicacao"],
                    batch_size=1000
                )
                
                # Converter para formato mais conveniente com validação adicional
                dados_treino_pd = pd.DataFrame([
                    {
                        'idUsuario': row['idUsuario'],
                        'historico': row['historico']
                    }
                    for row in dados_treino_list
                    if (row['historico'] is not None and 
                        isinstance(row['historico'], (list, np.ndarray)) and 
                        len(row['historico']) > 0)
                ])
                
                if len(dados_treino_pd) == 0:
                    raise ValueError("Nenhum dado de treino válido após processamento")
                
                dados_itens_pd = pd.DataFrame([
                    {
                        'conteudo_texto': row['conteudo_texto'] if row['conteudo_texto'] is not None else '',
                        'DataPublicacao': row['DataPublicacao']
                    }
                    for row in dados_itens_list
                ])
                
                logger.info(f"Dados coletados: {len(dados_treino_pd)} registros de treino válidos, {len(dados_itens_pd)} itens")
                
                # Adicionar estatísticas detalhadas
                usuarios_unicos = dados_treino_pd['idUsuario'].nunique()
                tamanho_medio_historico = dados_treino_pd['historico'].apply(len).mean()
                
                logger.info(f"Usuários únicos: {usuarios_unicos}")
                logger.info(f"Tamanho médio do histórico: {tamanho_medio_historico:.2f}")
                
                # Mostrar distribuição dos tamanhos de histórico
                tamanhos_historico = dados_treino_pd['historico'].apply(len)
                logger.info(f"Distribuição dos tamanhos de histórico:")
                logger.info(f"- Mínimo: {tamanhos_historico.min()}")
                logger.info(f"- Máximo: {tamanhos_historico.max()}")
                logger.info(f"- Mediana: {tamanhos_historico.median()}")
                
            except Exception as e:
                logger.error(f"Erro ao coletar dados do Spark: {str(e)}")
                raise

            # Processar features de conteúdo
            logger.info("Processando features de texto")
            features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)
            
            # Registrar métricas no MLflow se ativo
            if mlflow.active_run():
                metricas_pre_processamento = {
                    "n_usuarios_unicos": usuarios_unicos,
                    "tamanho_medio_historico": tamanho_medio_historico,
                    "n_itens_total": len(dados_itens_pd),
                    "dimensao_features": features_conteudo.shape[1]
                }
                self.mlflow_config.log_metricas(metricas_pre_processamento)

            # Criar mapeamentos em lotes
            logger.info("Criando mapeamentos usuário-item")
            self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)

            # Construir modelo
            n_usuarios = len(self.itens_usuario)
            n_itens = len(self.features_item)
            logger.info(f"Construindo modelo para {n_usuarios} usuários e {n_itens} itens")
            self.modelo = self._construir_modelo_neural(n_usuarios, n_itens)

            # Preparar dados de treino em lotes
            logger.info("Preparando dados para treinamento")
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd,
                features_conteudo,
                batch_size=1000
            )

            # Validar dados de treino
            if len(X_usuario) == 0:
                raise ValueError("Nenhum exemplo de treino gerado")

            logger.info(f"Dados de treino preparados: {len(X_usuario)} exemplos")
            logger.info(f"Distribuição de classes: {np.bincount(y)}")

            # Configurar callbacks
            callbacks = self._configurar_callbacks()

            # Treinar modelo
            logger.info("Iniciando treinamento do modelo neural")
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                epochs=5,
                batch_size=64,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            # Registrar métricas de treinamento no MLflow
            if mlflow.active_run():
                metricas_treinamento = {
                    "loss_final": historia.history['loss'][-1],
                    "accuracy_final": historia.history['accuracy'][-1],
                    "val_loss_final": historia.history['val_loss'][-1],
                    "val_accuracy_final": historia.history['val_accuracy'][-1],
                    "n_exemplos_treino": len(X_usuario),
                    "n_exemplos_positivos": np.sum(y == 1),
                    "n_exemplos_negativos": np.sum(y == 0)
                }
                self.mlflow_config.log_metricas(metricas_treinamento)

            logger.info("Treinamento concluído com sucesso")
            return historia

        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise   

    def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, batch_size=1000):
        """
        Prepara dados de treino em lotes para evitar sobrecarga de memória.
        """
        X_usuario, X_item, X_conteudo, y = [], [], [], []
        total_processado = 0
        
        for i in range(0, len(dados_treino_pd), batch_size):
            batch = dados_treino_pd.iloc[i:i+batch_size]
            
            for _, linha in batch.iterrows():
                usuario_id = linha['idUsuario']
                historico = linha['historico']
                
                if historico is None or len(historico) == 0:
                    continue
                
                # Amostras positivas
                for item_id in historico:
                    if item_id >= len(features_conteudo):
                        continue
                    X_usuario.append(usuario_id)
                    X_item.append(item_id)
                    X_conteudo.append(features_conteudo[item_id])
                    y.append(1)
                    total_processado += 1
                
                # Amostras negativas
                negativos = self._gerar_amostras_negativas(historico, len(features_conteudo))
                for item_id in negativos:
                    X_usuario.append(usuario_id)
                    X_item.append(item_id)
                    X_conteudo.append(features_conteudo[item_id])
                    y.append(0)
                    total_processado += 1
            
            logger.info(f"Processados {total_processado} exemplos de treino")
        
        if len(X_usuario) == 0:
            raise ValueError("Nenhum dado de treino válido encontrado")
        
        logger.info(f"Total de exemplos de treino preparados: {len(X_usuario)}")
        return (np.array(X_usuario), np.array(X_item), 
                np.array(X_conteudo), np.array(y))

    def _gerar_amostras_negativas(self, historico_positivo, n_itens, n_amostras=None):
        """
        Gera amostras negativas para treinamento.
        """
        if n_amostras is None:
            n_amostras = len(historico_positivo)
            
        todos_itens = set(range(n_itens))
        itens_negativos = list(todos_itens - set(historico_positivo))
        
        if not itens_negativos:
            logger.warning("Não há itens negativos disponíveis para amostragem")
            return []
        
        return np.random.choice(
            itens_negativos,
            size=min(n_amostras, len(itens_negativos)),
            replace=False
        )

    def prever(self, id_usuario, n_recomendacoes=10):
        """
        Gera recomendações para um usuário.
        """
        logger.info(f"Gerando previsões para usuário {id_usuario}")
        try:
            if id_usuario not in self.itens_usuario:
                return self._recomendacoes_usuario_novo()

            # Preparar dados
            todos_itens = list(self.features_item.keys())
            itens_nao_vistos = [i for i in todos_itens 
                               if i not in self.itens_usuario[id_usuario]]
            
            if not itens_nao_vistos:
                logger.warning("Usuário já viu todos os itens disponíveis")
                return []
            
            # Gerar previsões em lotes
            batch_size = 1024
            previsoes = []
            
            for i in range(0, len(itens_nao_vistos), batch_size):
                fim = min(i + batch_size, len(itens_nao_vistos))
                batch_itens = itens_nao_vistos[i:fim]
                
                usuario_input = np.array([id_usuario] * len(batch_itens))
                item_input = np.array(batch_itens)
                conteudo_input = np.array([
                    self.features_item[j]['vetor_conteudo'] 
                    for j in batch_itens
                ])

                batch_previsoes = self.modelo.predict(
                    [usuario_input, item_input, conteudo_input],
                    verbose=0
                )
                previsoes.extend(batch_previsoes.flatten())
                
                logger.info(f"Processados {fim} de {len(itens_nao_vistos)} itens para previsão")

            # Combinar scores e retornar top N
            scores = list(zip(itens_nao_vistos, previsoes))
            recomendacoes = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [item for item, _ in recomendacoes[:n_recomendacoes]]

        except Exception as e:
            logger.error(f"Erro ao gerar previsões: {str(e)}")
            raise

    def _recomendacoes_usuario_novo(self):
        """
        Recomendações para usuários novos.
        """
        logger.info("Gerando recomendações para usuário novo")
        try:
            # Ordenar por recência
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
        """
        logger.info(f"Salvando modelo em {caminho}")
        try:
            dados_modelo = {
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'itens_usuario': self.itens_usuario,
                'features_item': self.features_item,
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
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise