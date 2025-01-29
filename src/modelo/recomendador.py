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
        self.item_id_to_index = {}  # Mapeamento de ID string para índice
        self.index_to_item_id = {}  # Mapeamento de índice para ID string
        self.item_count = 0  # Contador para gerar índices únicos
        
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

    def _verificar_correspondencia_ids(self, dados_treino_pd, dados_itens_pd):
        """
        Verifica a correspondência entre IDs de itens nos históricos e no DataFrame de itens.
        """
        logger.info("Verificando correspondência de IDs")
        
        # Coletar todos os IDs de itens
        ids_itens = set(str(idx).strip() for idx in dados_itens_pd.index)
        
        # Coletar todos os IDs nos históricos
        ids_historicos = set()
        for historico in dados_treino_pd['historico']:
            if isinstance(historico, (list, np.ndarray)):
                ids_historicos.update(str(item).strip() for item in historico)
        
        # Análise
        ids_comuns = ids_itens.intersection(ids_historicos)
        ids_apenas_historico = ids_historicos - ids_itens
        ids_apenas_itens = ids_itens - ids_historicos
        
        logger.info(f"IDs comuns: {len(ids_comuns)}")
        logger.info(f"IDs apenas no histórico: {len(ids_apenas_historico)}")
        logger.info(f"IDs apenas nos itens: {len(ids_apenas_itens)}")
        
        if len(ids_apenas_historico) > 0:
            logger.warning("Exemplos de IDs no histórico mas não nos itens:")
            logger.warning(list(ids_apenas_historico)[:5])
        
        return ids_comuns, ids_apenas_historico, ids_apenas_itens

    def _validar_dados_entrada(self, dados_treino_pd, dados_itens_pd):
        """
        Valida os dados de entrada antes do processamento.
        """
        logger.info("Validando dados de entrada")
        
        # Validar dados de treino
        if dados_treino_pd.empty:
            raise ValueError("DataFrame de treino está vazio")
            
        # Verificar estrutura do histórico
        historicos_validos = dados_treino_pd['historico'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
        n_historicos_validos = historicos_validos.sum()
        
        logger.info(f"Total de usuários: {len(dados_treino_pd)}")
        logger.info(f"Usuários com histórico válido: {n_historicos_validos}")
        
        if n_historicos_validos == 0:
            raise ValueError("Nenhum usuário possui histórico válido")
        
        # Validar dados dos itens
        if dados_itens_pd.empty:
            raise ValueError("DataFrame de itens está vazio")
            
        # Verificar conteúdo texto
        textos_validos = dados_itens_pd['conteudo_texto'].notna()
        n_textos_validos = textos_validos.sum()
        
        logger.info(f"Total de itens: {len(dados_itens_pd)}")
        logger.info(f"Itens com texto válido: {n_textos_validos}")
        
        # Mostrar exemplos de dados
        logger.info("\nExemplo de dados:")
        logger.info("\nPrimeiros 5 itens do DataFrame de itens:")
        logger.info(dados_itens_pd.head())
        logger.info("\nPrimeiros 5 históricos:")
        for _, row in dados_treino_pd.head().iterrows():
            logger.info(f"Usuário: {row['idUsuario']}")
            logger.info(f"Histórico: {row['historico']}")
        
        return True

    def _verificar_dados_treino(self, dados_treino, dados_itens):
        """
        Verifica a estrutura dos dados antes do treinamento.
        """
        logger.info("Verificando estrutura dos dados de entrada")
        
        # Verificar DataFrame de treino
        logger.info("\nEstrutura do DataFrame de treino:")
        logger.info(dados_treino.printSchema())
        
        # Mostrar algumas estatísticas
        n_usuarios = dados_treino.select("idUsuario").distinct().count()
        n_historicos_vazios = dados_treino.filter(F.size("historico") == 0).count()
        
        logger.info(f"\nNúmero de usuários únicos: {n_usuarios}")
        logger.info(f"Número de históricos vazios: {n_historicos_vazios}")
        
        # Verificar alguns exemplos de histórico
        logger.info("\nExemplos de históricos:")
        dados_treino.select("idUsuario", "historico").show(5, truncate=False)
        
        # Verificar DataFrame de itens
        logger.info("\nEstrutura do DataFrame de itens:")
        logger.info(dados_itens.printSchema())
        
        return {
            "n_usuarios": n_usuarios,
            "n_historicos_vazios": n_historicos_vazios
        }

    def _diagnosticar_dados(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Função para diagnóstico detalhado dos dados.
        """
        logger.info("=== DIAGNÓSTICO DOS DADOS ===")
        
        # Análise do DataFrame de treino
        logger.info("\nDados de Treino:")
        logger.info(f"Shape: {dados_treino_pd.shape}")
        logger.info("\nPrimeiros registros:")
        logger.info(dados_treino_pd.head())
        logger.info("\nTipos de dados:")
        logger.info(dados_treino_pd.dtypes)
        
        # Análise dos históricos
        if 'historico' in dados_treino_pd.columns:
            historicos = dados_treino_pd['historico']
            logger.info("\nAnálise dos históricos:")
            logger.info(f"Número de históricos nulos: {historicos.isnull().sum()}")
            
            # Análise dos tipos de histórico
            tipos_historico = historicos.apply(type).value_counts()
            logger.info("\nTipos de histórico encontrados:")
            logger.info(tipos_historico)
            
            # Análise detalhada do primeiro histórico não nulo
            for hist in historicos:
                if hist is not None:
                    logger.info("\nExemplo de histórico:")
                    logger.info(f"Tipo: {type(hist)}")
                    logger.info(f"Conteúdo: {hist}")
                    logger.info(f"Primeiro item (se existir): {hist[0] if len(hist) > 0 else 'Vazio'}")
                    logger.info(f"Tipo do primeiro item: {type(hist[0]) if len(hist) > 0 else 'N/A'}")
                    break
            
            # Análise dos tamanhos de histórico
            tamanhos = historicos.apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            logger.info("\nEstatísticas de tamanho dos históricos:")
            logger.info(f"Média: {tamanhos.mean()}")
            logger.info(f"Mediana: {tamanhos.median()}")
            logger.info(f"Máximo: {tamanhos.max()}")
            logger.info(f"Mínimo: {tamanhos.min()}")
            logger.info(f"Históricos vazios: {(tamanhos == 0).sum()}")
        
        # Análise do DataFrame de itens
        logger.info("\nDados de Itens:")
        logger.info(f"Shape: {dados_itens_pd.shape}")
        logger.info("\nPrimeiros registros:")
        logger.info(dados_itens_pd.head())
        logger.info("\nÍndices dos itens:")
        logger.info(f"Primeiros 5 índices: {list(dados_itens_pd.index[:5])}")
        logger.info(f"Tipos dos índices: {type(dados_itens_pd.index)}")
        
        # Análise das features de conteúdo
        logger.info("\nFeatures de conteúdo:")
        logger.info(f"Shape: {features_conteudo.shape}")
        logger.info(f"Tipo: {type(features_conteudo)}")
        logger.info(f"Valores nulos: {np.isnan(features_conteudo).sum()}")
        
        return {
            "n_usuarios": len(dados_treino_pd),
            "n_itens": len(dados_itens_pd),
            "dim_features": features_conteudo.shape[1] if features_conteudo is not None else 0,
            "tipos_historico": tipos_historico.to_dict() if 'historico' in dados_treino_pd.columns else {},
            "estatisticas_tamanho_historico": {
                "media": tamanhos.mean(),
                "mediana": tamanhos.median(),
                "max": tamanhos.max(),
                "min": tamanhos.min(),
                "vazios": (tamanhos == 0).sum()
            } if 'historico' in dados_treino_pd.columns else {}
        }

    def _tentar_reconectar_spark(self, spark, max_tentativas=3):
        """
        Tenta reconectar ao Spark em caso de falha.
        """
        logger.info("Tentando reconectar ao Spark")
        for tentativa in range(max_tentativas):
            try:
                if spark._jsc is None or spark._jsc._sc is None:
                    spark = SparkSession.builder \
                        .appName("RecomendadorNoticias") \
                        .config("spark.driver.memory", "4g") \
                        .config("spark.executor.memory", "4g") \
                        .getOrCreate()
                
                spark.sql("SELECT 1").collect()
                logger.info("Reconexão com Spark bem-sucedida")
                return spark
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1} de reconexão falhou: {str(e)}")
                time.sleep(5)
        
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
                    time.sleep(2)
            
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

    def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos de usuários e itens com diagnóstico detalhado.
        """
        logger.info("Iniciando criação de mapeamentos")
        
        try:
            # Executar diagnóstico
            diagnostico = self._diagnosticar_dados(dados_treino_pd, dados_itens_pd, features_conteudo)
            
            # Primeiro, mapear os IDs dos itens do DataFrame de itens
            logger.info("Mapeando IDs dos itens")
            for coluna in dados_itens_pd.columns:
                if 'id' in coluna.lower():  # Procurar coluna que contenha 'id' no nome
                    logger.info(f"Usando coluna '{coluna}' como ID dos itens")
                    for _, linha in dados_itens_pd.iterrows():
                        item_id = str(linha[coluna]).strip()  # Usar o valor da coluna ID
                        if item_id not in self.item_id_to_index:
                            self.item_id_to_index[item_id] = self.item_count
                            self.index_to_item_id[self.item_count] = item_id
                            self.item_count += 1
                    break
            
            logger.info(f"Total de itens mapeados: {self.item_count}")
            logger.info(f"Exemplos de mapeamentos:")
            for item_id, idx in list(self.item_id_to_index.items())[:5]:
                logger.info(f"ID: {item_id} -> Índice: {idx}")
            
            # Processar usuários
            usuarios_processados = 0
            usuarios_validos = 0
            historicos_invalidos = 0
            
            for idx, linha in dados_treino_pd.iterrows():
                try:
                    historico = linha['historico']
                    usuario_id = linha['idUsuario']
                    
                    if historico is not None and isinstance(historico, (list, np.ndarray)):
                        # Converter IDs string para índices numéricos
                        historico_numerico = []
                        for item_id in historico:
                            item_id_str = str(item_id).strip()
                            if item_id_str in self.item_id_to_index:
                                indice = self.item_id_to_index[item_id_str]
                                if indice < len(features_conteudo):
                                    historico_numerico.append(indice)
                            else:
                                logger.warning(f"Item não encontrado no mapeamento: {item_id}")
                        
                        if historico_numerico:
                            self.itens_usuario[usuario_id] = set(historico_numerico)
                            usuarios_validos += 1
                        else:
                            historicos_invalidos += 1
                            logger.warning(f"Histórico sem itens válidos para usuário {usuario_id}")
                    else:
                        historicos_invalidos += 1
                        logger.warning(f"Histórico inválido para usuário {usuario_id}")
                    
                    usuarios_processados += 1
                    if usuarios_processados % 100 == 0:
                        logger.info(f"Progresso: {usuarios_processados}/{len(dados_treino_pd)} usuários processados")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
            
            # Log detalhado do processamento
            logger.info("\n=== RESUMO DO PROCESSAMENTO ===")
            logger.info(f"Total de itens únicos mapeados: {self.item_count}")
            logger.info(f"Total de usuários processados: {usuarios_processados}")
            logger.info(f"Usuários com histórico válido: {usuarios_validos}")
            logger.info(f"Históricos inválidos: {historicos_invalidos}")
            
            if usuarios_validos == 0:
                if len(dados_treino_pd) > 0:
                    primeiro_historico = dados_treino_pd.iloc[0]['historico']
                    logger.error(f"Exemplo de histórico problemático: {primeiro_historico}")
                    if isinstance(primeiro_historico, (list, np.ndarray)) and len(primeiro_historico) > 0:
                        logger.error(f"Primeiro item do histórico: {primeiro_historico[0]}")
                        logger.error(f"Tipo do primeiro item: {type(primeiro_historico[0])}")
                        
                        # Verificar se o item existe no mapeamento
                        item_id_str = str(primeiro_historico[0]).strip()
                        if item_id_str in self.item_id_to_index:
                            logger.error(f"Item encontrado no mapeamento com índice: {self.item_id_to_index[item_id_str]}")
                        else:
                            logger.error("Item não encontrado no mapeamento")
                            
                    # Mostrar alguns exemplos do mapeamento
                    logger.error("Primeiros 5 mapeamentos:")
                    for item_id, idx in list(self.item_id_to_index.items())[:5]:
                        logger.error(f"ID: {item_id} -> Índice: {idx}")
                
                raise ValueError("Nenhum usuário com histórico válido após processamento")
            
            # Processar features dos itens
            features_item_novo = {}
            for item_id, indice in self.item_id_to_index.items():
                if indice < len(features_conteudo):
                    features_item_novo[indice] = {
                        'vetor_conteudo': features_conteudo[indice],
                        'timestamp': 0  # Usar timestamp padrão se não disponível
                    }
            
            self.features_item = features_item_novo
            logger.info(f"Processados {len(self.features_item)} itens com features")
            
        except Exception as e:
            logger.error(f"Erro durante criação de mapeamentos: {str(e)}")
            raise

    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo com melhor gestão de recursos e reconexão automática.
        """
        logger.info("Iniciando treinamento")
        try:
            # Verificar dados iniciais
            estatisticas = self._verificar_dados_treino(dados_treino, dados_itens)
            logger.info("Estatísticas iniciais dos dados:")
            logger.info(estatisticas)

            # Amostra dos dados para debug
            logger.info("Amostra dos dados de treino:")
            logger.info(dados_treino.select("idUsuario", "historico").show(5, truncate=False))
            
            logger.info("Amostra dos dados de itens:")
            logger.info(dados_itens.select("*").show(5, truncate=False))

            # Verificar se os DataFrames estão vazios
            if dados_treino.count() == 0:
                raise ValueError("DataFrame de treino está vazio")
                
            if dados_itens.count() == 0:
                raise ValueError("DataFrame de itens está vazio")
                
            # Verificar se há históricos válidos
            dados_treino = dados_treino.withColumn(
                "historico_valido",
                F.size(F.col("historico")) > 0
            )
            
            n_historicos_validos = dados_treino.filter(
                F.col("historico_valido")
            ).count()
            
            if n_historicos_validos == 0:
                raise ValueError("Nenhum usuário com histórico válido encontrado")
                
            logger.info(f"Encontrados {n_historicos_validos} usuários com histórico válido")

            # Coletar dados do Spark de forma segura
            logger.info("Coletando dados do Spark")
            try:
                # Filtrar apenas registros válidos antes de coletar
                dados_treino_filtrado = dados_treino.filter(
                    F.size(F.col('historico')) > 0
                )
                
                # Coletar dados em lotes
                dados_treino_list = []
                batch_size = 1000
                
                # Usar coalesce para otimizar a coleta
                dados_treino_filtrado = dados_treino_filtrado.coalesce(1)
                
                for offset in range(0, n_historicos_validos, batch_size):
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
                
                # Verificar correspondência de IDs antes de prosseguir
                self._verificar_correspondencia_ids(dados_treino_pd, dados_itens_pd)

                # Processar features de conteúdo
                logger.info("Processando features de texto")
                features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)
                
                # Registrar métricas no MLflow se ativo
                if mlflow.active_run():
                    metricas_pre_processamento = {
                        "n_usuarios_unicos": dados_treino_pd['idUsuario'].nunique(),
                        "n_itens_total": len(dados_itens_pd),
                        "dimensao_features": features_conteudo.shape[1]
                    }
                    self.mlflow_config.log_metricas(metricas_pre_processamento)

                # Criar mapeamentos
                logger.info("Criando mapeamentos usuário-item")
                self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)

                # Construir modelo
                n_usuarios = len(self.itens_usuario)
                n_itens = len(self.features_item)
                logger.info(f"Construindo modelo para {n_usuarios} usuários e {n_itens} itens")
                self.modelo = self._construir_modelo_neural(n_usuarios, n_itens)

                # Preparar dados de treino
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
                
        except Exception as e:
            logger.error(f"Erro ao coletar dados do Spark: {str(e)}")
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
                'dim_features_texto': self.dim_features_texto,
                'item_id_to_index': self.item_id_to_index,
                'index_to_item_id': self.index_to_item_id,
                'item_count': self.item_count
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
            instancia.item_id_to_index = dados_modelo['item_id_to_index']
            instancia.index_to_item_id = dados_modelo['index_to_item_id']
            instancia.item_count = dados_modelo['item_count']
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization
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
from pyspark.sql.functions import year, expr
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.constraints import MaxNorm

try:
    nltk.download('stopwords')
except:
    pass
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        """
        Inicializa o recomendador híbrido.

        Args:
            dim_embedding: Dimensão dos embeddings para usuários e itens
            dim_features_texto: Dimensão das features de texto
            mlflow_config: Configuração do MLflow
        """
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        self.item_id_to_index = {}  # Mapeamento de URL para índice
        self.index_to_item_id = {}  # Mapeamento de índice para URL
        self.item_count = 0
        
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

    def _criar_features_conteudo_pandas(self, dados_itens_pd):
        """
        Cria features de conteúdo usando TF-IDF.

        Args:
            dados_itens_pd: DataFrame pandas com os dados dos itens

        Returns:
            np.array: Matrix de features TF-IDF
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

    def _verificar_correspondencia_ids(self, dados_treino_pd, dados_itens_pd):
        """
        Verifica a correspondência entre URLs nos históricos e no DataFrame de itens.

        Args:
            dados_treino_pd: DataFrame pandas com dados de treino
            dados_itens_pd: DataFrame pandas com dados dos itens

        Returns:
            tuple: Conjuntos de URLs comuns, apenas no histórico e apenas nos itens
        """
        logger.info("Verificando correspondência de URLs")
        
        # Coletar todas as URLs de itens
        urls_itens = set(str(page).strip() for page in dados_itens_pd['page'] if page)
        
        # Coletar todas as URLs nos históricos
        urls_historicos = set()
        for historico in dados_treino_pd['historico']:
            if isinstance(historico, (list, np.ndarray)):
                urls_historicos.update(str(url).strip() for url in historico if url)
        
        # Análise
        urls_comuns = urls_itens.intersection(urls_historicos)
        urls_apenas_historico = urls_historicos - urls_itens
        urls_apenas_itens = urls_itens - urls_historicos
        
        logger.info(f"URLs comuns: {len(urls_comuns)}")
        logger.info(f"URLs apenas no histórico: {len(urls_apenas_historico)}")
        logger.info(f"URLs apenas nos itens: {len(urls_apenas_itens)}")
        
        if len(urls_apenas_historico) > 0:
            logger.warning("Exemplos de URLs no histórico mas não nos itens:")
            logger.warning(list(urls_apenas_historico)[:5])
        
        return urls_comuns, urls_apenas_historico, urls_apenas_itens

    def _validar_dados_entrada(self, dados_treino_pd, dados_itens_pd):
        """
        Valida os dados de entrada antes do processamento.

        Args:
            dados_treino_pd: DataFrame pandas com dados de treino
            dados_itens_pd: DataFrame pandas com dados dos itens

        Returns:
            bool: True se os dados são válidos
        """
        logger.info("Validando dados de entrada")
        
        # Validar dados de treino
        if dados_treino_pd.empty:
            raise ValueError("DataFrame de treino está vazio")
        
        # Verificar colunas necessárias
        colunas_necessarias_treino = ['idUsuario', 'historico']
        colunas_necessarias_itens = ['page', 'conteudo_texto']
        
        for coluna in colunas_necessarias_treino:
            if coluna not in dados_treino_pd.columns:
                raise ValueError(f"Coluna {coluna} não encontrada nos dados de treino")
                
        for coluna in colunas_necessarias_itens:
            if coluna not in dados_itens_pd.columns:
                raise ValueError(f"Coluna {coluna} não encontrada nos dados de itens")
            
        # Verificar estrutura do histórico
        historicos_validos = dados_treino_pd['historico'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
        n_historicos_validos = historicos_validos.sum()
        
        logger.info(f"Total de usuários: {len(dados_treino_pd)}")
        logger.info(f"Usuários com histórico válido: {n_historicos_validos}")
        
        if n_historicos_validos == 0:
            raise ValueError("Nenhum usuário possui histórico válido")
        
        # Validar dados dos itens
        if dados_itens_pd.empty:
            raise ValueError("DataFrame de itens está vazio")
            
        # Verificar conteúdo texto
        textos_validos = dados_itens_pd['conteudo_texto'].notna()
        n_textos_validos = textos_validos.sum()
        
        logger.info(f"Total de itens: {len(dados_itens_pd)}")
        logger.info(f"Itens com texto válido: {n_textos_validos}")
        
        # Verificar URLs únicas
        n_urls_unicas = dados_itens_pd['page'].nunique()
        logger.info(f"URLs únicas nos itens: {n_urls_unicas}")
        
        # Verificar datas
        if 'DataPublicacao' in dados_itens_pd.columns:
            datas_validas = dados_itens_pd['DataPublicacao'].notna()
            n_datas_validas = datas_validas.sum()
            logger.info(f"Itens com datas válidas: {n_datas_validas}")
            
            if n_datas_validas > 0:
                data_min = dados_itens_pd['DataPublicacao'].min()
                data_max = dados_itens_pd['DataPublicacao'].max()
                logger.info(f"Range de datas: {data_min} até {data_max}")
        
        # Mostrar exemplos de dados
        logger.info("\nExemplo de dados:")
        logger.info("\nPrimeiros 5 itens do DataFrame de itens:")
        logger.info(dados_itens_pd[['page', 'conteudo_texto']].head())
        logger.info("\nPrimeiros 5 históricos:")
        for _, row in dados_treino_pd.head().iterrows():
            logger.info(f"Usuário: {row['idUsuario']}")
            logger.info(f"Histórico: {row['historico'][:5] if len(row['historico']) > 5 else row['historico']}")
        
        return True

    def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos entre URLs e índices numéricos para itens e usuários.

        Args:
            dados_treino_pd: DataFrame pandas com dados de treino
            dados_itens_pd: DataFrame pandas com dados dos itens
            features_conteudo: Matrix numpy com features TF-IDF dos itens
        """
        logger.info("Iniciando criação de mapeamentos...")
        
        try:
            # Mapear itens usando a coluna 'page' dos dados de itens
            logger.info("Mapeando páginas para índices")
            
            # Reset index para garantir alinhamento com features
            dados_itens_pd = dados_itens_pd.reset_index(drop=True)
            
            for idx, page in enumerate(dados_itens_pd['page']):
                if page and isinstance(page, str):
                    page = page.strip()
                    self.item_id_to_index[page] = idx
                    self.index_to_item_id[idx] = page
                    self.item_count = max(self.item_count, idx + 1)
            
            logger.info(f"Total de itens mapeados: {self.item_count}")
            logger.info("Exemplos de mapeamentos:")
            for page, idx in list(self.item_id_to_index.items())[:5]:
                logger.info(f"URL: {page} -> Índice: {idx}")
            
            # Processar usuários e seus históricos
            usuarios_processados = 0
            usuarios_validos = 0
            historicos_invalidos = 0
            
            # Criar um mapeamento de usuário para índice
            usuarios_unicos = dados_treino_pd['idUsuario'].unique()
            usuario_para_idx = {usuario: idx for idx, usuario in enumerate(usuarios_unicos)}
            
            for _, linha in dados_treino_pd.iterrows():
                try:
                    historico = linha['historico']
                    usuario_id = linha['idUsuario']
                    
                    if historico is not None and isinstance(historico, (list, np.ndarray)):
                        # Converter URLs para índices numéricos
                        historico_numerico = []
                        for url in historico:
                            url_str = str(url).strip()
                            if url_str in self.item_id_to_index:
                                indice = self.item_id_to_index[url_str]
                                if indice < len(features_conteudo):
                                    historico_numerico.append(indice)
                            else:
                                logger.warning(f"URL não encontrada no mapeamento: {url_str}")
                        
                        if historico_numerico:
                            self.itens_usuario[usuario_para_idx[usuario_id]] = set(historico_numerico)
                            usuarios_validos += 1
                        else:
                            historicos_invalidos += 1
                            logger.warning(f"Histórico sem itens válidos para usuário {usuario_id}")
                    
                    usuarios_processados += 1
                    if usuarios_processados % 100 == 0:
                        logger.info(f"Progresso: {usuarios_processados}/{len(dados_treino_pd)} usuários processados")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
            
            # Log do processamento
            logger.info("\n=== RESUMO DO PROCESSAMENTO ===")
            logger.info(f"Total de itens únicos mapeados: {self.item_count}")
            logger.info(f"Total de usuários processados: {usuarios_processados}")
            logger.info(f"Usuários com histórico válido: {usuarios_validos}")
            logger.info(f"Históricos inválidos: {historicos_invalidos}")
            
            if usuarios_validos == 0:
                raise ValueError("Nenhum usuário com histórico válido após processamento")
            
            # Processar features dos itens
            self.features_item = {}
            for page, idx in self.item_id_to_index.items():
                if idx < len(features_conteudo):
                    self.features_item[idx] = {
                        'vetor_conteudo': features_conteudo[idx],
                        'timestamp': 0  # Usar timestamp padrão se não disponível
                    }
            
            logger.info(f"Processados {len(self.features_item)} itens com features")
            
        except Exception as e:
            logger.error(f"Erro durante criação de mapeamentos: {str(e)}")
            raise

    '''def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, batch_size=1000):
        """
        Prepara os dados de treino em lotes para evitar sobrecarga de memória.

        Args:
            dados_treino_pd: DataFrame pandas com dados de treino
            features_conteudo: Matrix numpy com features TF-IDF dos itens
            batch_size: Tamanho do lote para processamento

        Returns:
            tuple: Arrays numpy com dados de treino (X_usuario, X_item, X_conteudo, y)
        """
        logger.info("Preparando dados de treino em lotes")
        
        X_usuario_list = []
        X_item_list = []
        X_conteudo_list = []
        y_list = []
        
        try:
            # Processar cada usuário
            for usuario_idx, historico in self.itens_usuario.items():
                # Gerar exemplos positivos
                for item_idx in historico:
                    if item_idx in self.features_item:
                        X_usuario_list.append(usuario_idx)
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(1)
                
                # Gerar exemplos negativos
                todos_itens = set(self.features_item.keys())
                itens_negativos = todos_itens - historico
                
                # Amostrar aleatoriamente o mesmo número de exemplos negativos
                n_negativos = min(len(historico), len(itens_negativos))
                itens_negativos_amostra = np.random.choice(list(itens_negativos), n_negativos, replace=False)
                
                for item_idx in itens_negativos_amostra:
                    X_usuario_list.append(usuario_idx)
                    X_item_list.append(item_idx)
                    X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                    y_list.append(0)
                
                # Processar em lotes se necessário
                if len(X_usuario_list) >= batch_size:
                    logger.info(f"Processados {len(X_usuario_list)} exemplos")
            
            # Converter para arrays numpy
            X_usuario = np.array(X_usuario_list)
            X_item = np.array(X_item_list)
            X_conteudo = np.array(X_conteudo_list)
            y = np.array(y_list)
            
            # Log de estatísticas
            logger.info(f"Total de exemplos gerados: {len(y)}")
            logger.info(f"Exemplos positivos: {np.sum(y == 1)}")
            logger.info(f"Exemplos negativos: {np.sum(y == 0)}")
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise'''
    
    def _mostrar_exemplo_detalhado(self, usuario_idx, item_idx, label):
        """
        Mostra detalhes de um exemplo de treino.
        
        Args:
            usuario_idx: ID do usuário
            item_idx: ID do item
            label: 1 para exemplo positivo, 0 para negativo
        """
        try:
            url_original = self.index_to_item_id[item_idx]
            features = self.features_item[item_idx]['vetor_conteudo']
            
            logger.info(f"""
        ======= EXEMPLO {'POSITIVO' if label == 1 else 'NEGATIVO'} =======
        1. Informações Básicas:
        - ID do Usuário: {usuario_idx}
        - ID do Item: {item_idx}
        - Label: {label} ({'interagiu' if label == 1 else 'não interagiu'})
        
        2. URL do Artigo: 
        {url_original}
        
        3. Features do Conteúdo (primeiros 5 valores):
        {features[:5]}
        
        4. Formato dos Dados:
        - X_usuario: {usuario_idx} (tipo: {type(usuario_idx)})
        - X_item: {item_idx} (tipo: {type(item_idx)})
        - X_conteudo: array de tamanho {len(features)} (tipo: {type(features)})
        - y: {label} (tipo: {type(label)})
        """)
        except Exception as e:
            logger.error(f"Erro ao mostrar exemplo detalhado: {str(e)}")

    def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, 
                                    max_exemplos_total=1000000,
                                    max_exemplos_por_usuario=100,
                                    batch_size=10000):
        """
        Prepara os dados de treino em lotes, mostrando exemplos detalhados.
        
        Args:
            dados_treino_pd: DataFrame pandas com dados de treino
            features_conteudo: Matrix numpy com features TF-IDF
            max_exemplos_total: Número máximo total de exemplos
            max_exemplos_por_usuario: Número máximo de exemplos por usuário
            batch_size: Tamanho do lote para processamento
        
        Returns:
            tuple: (X_usuario, X_item, X_conteudo, y) arrays numpy
        """
        logger.info("Preparando dados de treino em lotes")
        
        X_usuario_list = []
        X_item_list = []
        X_conteudo_list = []
        y_list = []
        
        try:
            # Mostrar exemplos detalhados do primeiro usuário
            primeiro_usuario = list(self.itens_usuario.keys())[0]
            historico_primeiro_usuario = self.itens_usuario[primeiro_usuario]
            
            logger.info("\n=== EXEMPLOS DETALHADOS DO PRIMEIRO USUÁRIO ===")
            logger.info(f"ID do Usuário: {primeiro_usuario}")
            logger.info(f"Tamanho do histórico: {len(historico_primeiro_usuario)}")
            
            # Mostrar exemplo positivo
            if historico_primeiro_usuario:
                primeiro_item_positivo = list(historico_primeiro_usuario)[0]
                self._mostrar_exemplo_detalhado(primeiro_usuario, primeiro_item_positivo, 1)
                
                # Adicionar exemplo positivo aos dados
                X_usuario_list.append(primeiro_usuario)
                X_item_list.append(primeiro_item_positivo)
                X_conteudo_list.append(self.features_item[primeiro_item_positivo]['vetor_conteudo'])
                y_list.append(1)
            
            # Mostrar exemplo negativo
            todos_itens = set(self.features_item.keys())
            itens_nao_lidos = todos_itens - historico_primeiro_usuario
            if itens_nao_lidos:
                primeiro_item_negativo = list(itens_nao_lidos)[0]
                self._mostrar_exemplo_detalhado(primeiro_usuario, primeiro_item_negativo, 0)
                
                # Adicionar exemplo negativo aos dados
                X_usuario_list.append(primeiro_usuario)
                X_item_list.append(primeiro_item_negativo)
                X_conteudo_list.append(self.features_item[primeiro_item_negativo]['vetor_conteudo'])
                y_list.append(0)
            
            exemplos_processados = len(y_list)
            logger.info(f"\nProcessando demais usuários...")
            
            # Processar demais usuários
            for usuario_idx, historico in list(self.itens_usuario.items())[1:]:
                if exemplos_processados >= max_exemplos_total:
                    logger.info(f"Atingido limite máximo de exemplos: {max_exemplos_total}")
                    break
                    
                # Limitar exemplos por usuário
                n_exemplos_usuario = min(len(historico), max_exemplos_por_usuario // 2)
                
                # Amostrar histórico se necessário
                historico_amostrado = list(historico)
                if len(historico_amostrado) > n_exemplos_usuario:
                    historico_amostrado = np.random.choice(
                        historico_amostrado, 
                        n_exemplos_usuario, 
                        replace=False
                    ).tolist()
                
                # Exemplos positivos
                for item_idx in historico_amostrado:
                    if item_idx in self.features_item:
                        X_usuario_list.append(usuario_idx)
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(1)
                        exemplos_processados += 1
                
                # Exemplos negativos
                itens_negativos = todos_itens - historico
                n_negativos = len(historico_amostrado)
                
                if len(itens_negativos) > n_negativos:
                    itens_negativos = np.random.choice(
                        list(itens_negativos), 
                        n_negativos, 
                        replace=False
                    ).tolist()
                
                for item_idx in itens_negativos:
                    X_usuario_list.append(usuario_idx)
                    X_item_list.append(item_idx)
                    X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                    y_list.append(0)
                    exemplos_processados += 1
                
                # Log de progresso
                if exemplos_processados % batch_size == 0:
                    logger.info(f"Processados {exemplos_processados} exemplos")
            
            # Converter para arrays numpy
            X_usuario = np.array(X_usuario_list)
            X_item = np.array(X_item_list)
            X_conteudo = np.array(X_conteudo_list)
            y = np.array(y_list)
            
            # Estatísticas finais
            logger.info("\n=== ESTATÍSTICAS FINAIS ===")
            logger.info(f"Total de exemplos gerados: {len(y)}")
            logger.info(f"Exemplos positivos: {np.sum(y == 1)}")
            logger.info(f"Exemplos negativos: {np.sum(y == 0)}")
            logger.info(f"\nFormato dos arrays:")
            logger.info(f"X_usuario: {X_usuario.shape}")
            logger.info(f"X_item: {X_item.shape}")
            logger.info(f"X_conteudo: {X_conteudo.shape}")
            logger.info(f"y: {y.shape}")
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise

    '''def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido.
        
        Args:
            n_usuarios: Número total de usuários
            n_itens: Número total de itens
            
        Returns:
            Model: Modelo Keras compilado
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Input layers
            entrada_usuario = Input(shape=(1,), name='input_usuario')
            entrada_item = Input(shape=(1,), name='input_item')
            entrada_conteudo = Input(shape=(self.dim_features_texto,), name='input_conteudo')

            # Embedding layers
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=self.dim_embedding,
                name='embedding_usuario'
            )(entrada_usuario)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=self.dim_embedding,
                name='embedding_item'
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten(name='flatten_usuario')(embedding_usuario)
            item_flat = Flatten(name='flatten_item')(embedding_item)

            # Concatenate all features
            concat = Concatenate(name='concatenate')([
                usuario_flat,
                item_flat,
                entrada_conteudo
            ])

            # Dense layers
            denso1 = Dense(128, activation='relu', name='dense_1')(concat)
            dropout1 = Dropout(0.3, name='dropout_1')(denso1)
            
            denso2 = Dense(64, activation='relu', name='dense_2')(dropout1)
            dropout2 = Dropout(0.2, name='dropout_2')(denso2)
            
            # Output layer
            saida = Dense(1, activation='sigmoid', name='output')(dropout2)

            # Create model
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida,
                name='modelo_hibrido'
            )
            
            # Compile model
            modelo.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )
            
            # Log model summary
            logger.info("Arquitetura do modelo:")
            modelo.summary(print_fn=logger.info)
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise'''
    
    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido com forte regularização para reduzir overfitting.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Configurações de regularização
            embedding_dim = 16
            dense_units = [32, 16]  # Unidades nas camadas densas
            dropout_rates = [0.4, 0.3]  # Taxas de dropout mais agressivas
            l2_factors = {
                'embedding': 0.02,
                'dense': 0.01,
                'output': 0.01
            }
            
            # Input layers
            entrada_usuario = Input(shape=(1,), name='input_usuario')
            entrada_item = Input(shape=(1,), name='input_item')
            entrada_conteudo = Input(shape=(self.dim_features_texto,), name='input_conteudo')

            # Embedding com forte regularização
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_factors['embedding']),
                embeddings_constraint=tf.keras.constraints.MaxNorm(2.0),
                name='embedding_usuario'
            )(entrada_usuario)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_factors['embedding']),
                embeddings_constraint=tf.keras.constraints.MaxNorm(2.0),
                name='embedding_item'
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten(name='flatten_usuario')(embedding_usuario)
            item_flat = Flatten(name='flatten_item')(embedding_item)

            # Redução de dimensionalidade do conteúdo com regularização
            conteudo_dense = Dense(
                32,
                activation='relu',
                kernel_regularizer=l2(l2_factors['dense']),
                kernel_constraint=tf.keras.constraints.MaxNorm(2.0),
                name='conteudo_reduction'
            )(entrada_conteudo)
            conteudo_norm = BatchNormalization(name='batch_norm_conteudo')(conteudo_dense)
            conteudo_drop = Dropout(0.3, name='dropout_conteudo')(conteudo_norm)

            # Concatenação com normalização
            concat = Concatenate(name='concatenate')([
                usuario_flat,
                item_flat,
                conteudo_drop
            ])
            concat_norm = BatchNormalization(name='batch_norm_concat')(concat)
            
            # Primeira camada densa
            dense1 = Dense(
                dense_units[0],
                activation='relu',
                kernel_regularizer=l2(l2_factors['dense']),
                kernel_constraint=tf.keras.constraints.MaxNorm(2.0),
                activity_regularizer=l1(0.01),
                name='dense_1'
            )(concat_norm)
            batch1 = BatchNormalization(name='batch_norm_1')(dense1)
            drop1 = Dropout(dropout_rates[0], name='dropout_1')(batch1)
            
            # Segunda camada densa
            dense2 = Dense(
                dense_units[1],
                activation='relu',
                kernel_regularizer=l2(l2_factors['dense']),
                kernel_constraint=tf.keras.constraints.MaxNorm(2.0),
                activity_regularizer=l1(0.01),
                name='dense_2'
            )(drop1)
            batch2 = BatchNormalization(name='batch_norm_2')(dense2)
            drop2 = Dropout(dropout_rates[1], name='dropout_2')(batch2)

            # Camada de saída com regularização
            saida = Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=l2(l2_factors['output']),
                kernel_constraint=tf.keras.constraints.MaxNorm(1.0),
                name='output'
            )(drop2)

            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida,
                name='modelo_hibrido_regularizado'
            )
            
            # Otimizador com clipping de gradientes
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0,  # Clipping de gradientes
                clipvalue=0.5
            )
            
            # Compilar modelo
            modelo.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),  # Label smoothing
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise


    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo com os dados fornecidos.

        Args:
            dados_treino: DataFrame Spark com dados de treino
            dados_itens: DataFrame Spark com dados dos itens

        Returns:
            History: Histórico de treinamento do modelo
        """
        logger.info("Iniciando treinamento do modelo")
        try:
            # Definir limites de datas
            max_year = 2030
            min_year = 1970
            
            # Filtrar datas válidas antes da conversão
            dados_itens = dados_itens.filter(
                (F.year(F.col("DataPublicacao")) >= min_year) &
                (F.year(F.col("DataPublicacao")) <= max_year)
            )
            
            # Converter dados Spark para pandas com tratamento de erro
            try:
                logger.info("Convertendo dados Spark para pandas")
                dados_treino_pd = dados_treino.toPandas()
                dados_itens_pd = dados_itens.toPandas()
                
                logger.info(f"Dados convertidos - Treino: {dados_treino_pd.shape}, Itens: {dados_itens_pd.shape}")
            except ValueError as e:
                if "year" in str(e) and "out of range" in str(e):
                    logger.error("Detectadas datas inválidas nos dados")
                    raise ValueError("Datas inválidas detectadas nos dados. Por favor, verifique o intervalo de datas.")
                raise
            
            # Validar dados
            self._validar_dados_entrada(dados_treino_pd, dados_itens_pd)
            
            # Verificar correspondência de IDs
            self._verificar_correspondencia_ids(dados_treino_pd, dados_itens_pd)
            
            # Criar features de conteúdo
            features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)
            
            # Criar mapeamentos
            self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)
            
            # Preparar dados de treino
            logger.info("Preparando dados para treinamento")
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd, 
                features_conteudo
            )
            
            if len(X_usuario) == 0:
                raise ValueError("Nenhum exemplo de treino gerado")
            
            logger.info(f"Dados de treino preparados: {len(X_usuario)} exemplos")
            logger.info(f"Distribuição de classes: {np.bincount(y)}")
            
            # Construir modelo
            self.modelo = self._construir_modelo_neural(
                len(self.itens_usuario),
                self.item_count
            )
            
            # Configurar callbacks
            callbacks = [
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
            
            # Treinar modelo
            logger.info("Iniciando treinamento do modelo neural")
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                validation_split=0.2,
                epochs=10,
                batch_size=64,
                callbacks=callbacks,
                verbose=1
            )
            
            # Registrar métricas no MLflow
            if mlflow.active_run():
                metricas = {
                    "loss_final": historia.history['loss'][-1],
                    "val_loss_final": historia.history['val_loss'][-1],
                    "accuracy_final": historia.history['accuracy'][-1],
                    "val_accuracy_final": historia.history['val_accuracy'][-1],
                    "n_usuarios": len(self.itens_usuario),
                    "n_itens": self.item_count,
                    "n_exemplos_treino": len(X_usuario),
                    "n_exemplos_positivos": np.sum(y == 1),
                    "n_exemplos_negativos": np.sum(y == 0)
                }
                self.mlflow_config.log_metricas(metricas)
            
            logger.info("Treinamento concluído com sucesso")
            return historia
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise

    def prever(self, usuario_id, candidatos=None, k=10):
        """
        Faz previsões para um usuário.

        Args:
            usuario_id: ID do usuário
            candidatos: Lista opcional de IDs de itens candidatos
            k: Número de recomendações a retornar

        Returns:
            list: Lista de IDs dos itens recomendados
        """
        if not self.modelo:
            raise ValueError("Modelo não treinado")
            
        if usuario_id not in self.itens_usuario:
            logger.warning(f"Usuário {usuario_id} não encontrado no conjunto de treino")
            return []
            
        if not candidatos:
            candidatos = list(self.features_item.keys())
        
        # Preparar dados para previsão
        X_usuario = np.array([usuario_id] * len(candidatos))
        X_item = np.array(candidatos)
        X_conteudo = np.array([self.features_item[idx]['vetor_conteudo'] for idx in candidatos])
        
        # Fazer previsões
        previsoes = self.modelo.predict([X_usuario, X_item, X_conteudo])
        
        # Ordenar e retornar os top-k itens
        indices_ordenados = np.argsort(previsoes.flatten())[::-1][:k]
        return [self.index_to_item_id[candidatos[i]] for i in indices_ordenados]

    def salvar_modelo(self, caminho):
        """
        Salva o modelo em disco.
        """
        logger.info(f"Salvando modelo em {caminho}")
        try:
            dados_modelo = {
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'item_id_to_index': self.item_id_to_index,
                'index_to_item_id': self.index_to_item_id,
                'item_count': self.item_count,
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
        Carrega um modelo salvo do disco.

        Args:
            caminho: Caminho do modelo salvo

        Returns:
            RecomendadorHibrido: Instância carregada do modelo
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
            instancia.item_id_to_index = dados_modelo['item_id_to_index']
            instancia.index_to_item_id = dados_modelo['index_to_item_id']
            instancia.item_count = dados_modelo['item_count']
            instancia.itens_usuario = dados_modelo['itens_usuario']
            instancia.features_item = dados_modelo['features_item']
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise