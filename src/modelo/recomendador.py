'''import numpy as np
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
        Mantém os IDs originais dos usuários.

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
            
            for _, linha in dados_treino_pd.iterrows():
                try:
                    historico = linha['historico']
                    usuario_id = linha['idUsuario']  # Usar ID original
                    
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
                            self.itens_usuario[usuario_id] = set(historico_numerico)  # Usar ID original
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
            
            # Mostrar exemplos de IDs de usuários
            logger.info("\nExemplos de IDs de usuários mantidos:")
            for usuario_id in list(self.itens_usuario.keys())[:5]:
                logger.info(f"ID original: {usuario_id}")
            
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
        Mantém os IDs originais dos usuários.
        """
        logger.info("Preparando dados de treino em lotes")
        
        X_usuario_list = []
        X_item_list = []
        X_conteudo_list = []
        y_list = []
        
        try:
            # Mostrar exemplos detalhados do primeiro usuário
            primeiro_usuario = list(self.itens_usuario.keys())[0]  # ID original
            historico_primeiro_usuario = self.itens_usuario[primeiro_usuario]
            
            logger.info("\n=== EXEMPLOS DETALHADOS DO PRIMEIRO USUÁRIO ===")
            logger.info(f"ID do Usuário (original): {primeiro_usuario}")
            logger.info(f"Tamanho do histórico: {len(historico_primeiro_usuario)}")
            
            # Mostrar exemplo positivo
            if historico_primeiro_usuario:
                primeiro_item_positivo = list(historico_primeiro_usuario)[0]
                self._mostrar_exemplo_detalhado(primeiro_usuario, primeiro_item_positivo, 1)
                
                # Adicionar exemplo positivo aos dados
                X_usuario_list.append(primeiro_usuario)  # ID original
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
                X_usuario_list.append(primeiro_usuario)  # ID original
                X_item_list.append(primeiro_item_negativo)
                X_conteudo_list.append(self.features_item[primeiro_item_negativo]['vetor_conteudo'])
                y_list.append(0)
            
            exemplos_processados = len(y_list)
            logger.info(f"\nProcessando demais usuários...")
            
            # Processar demais usuários
            for usuario_id, historico in list(self.itens_usuario.items())[1:]:  # usuario_id é o ID original
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
                        X_usuario_list.append(usuario_id)  # ID original
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
                    X_usuario_list.append(usuario_id)  # ID original
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
            
            # Mostrar alguns exemplos dos IDs originais
            logger.info("\nExemplos de IDs de usuários nos dados de treino:")
            unique_users = np.unique(X_usuario)
            logger.info(f"Número de usuários únicos: {len(unique_users)}")
            logger.info("Primeiros 5 IDs de usuários:")
            for user_id in unique_users[:5]:
                logger.info(f"- {user_id}")
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise    

    
    
    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido com regularização para reduzir overfitting.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Configurações
            embedding_dim = 16
            dense_units = [32, 16]
            dropout_rate = 0.3
            l2_lambda = 0.01
            
            # Input layers
            entrada_usuario = Input(shape=(1,))
            entrada_item = Input(shape=(1,))
            entrada_conteudo = Input(shape=(self.dim_features_texto,))

            # Embedding layers
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda)
            )(entrada_usuario)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda)
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten()(embedding_usuario)
            item_flat = Flatten()(embedding_item)

            # Concatenação
            concat = Concatenate()([
                usuario_flat,
                item_flat,
                entrada_conteudo
            ])
            
            # Primeira camada densa
            x = Dense(
                dense_units[0],
                activation='relu',
                kernel_regularizer=l2(l2_lambda)
            )(concat)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Segunda camada densa
            x = Dense(
                dense_units[1],
                activation='relu',
                kernel_regularizer=l2(l2_lambda)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            # Camada de saída
            saida = Dense(1, activation='sigmoid')(x)

            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida
            )
            
            # Otimizador com apenas clipnorm
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0  # Usando apenas clipnorm
            )
            
            modelo.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ]
            )
            
            # Log da arquitetura
            logger.info("\nArquitetura do modelo:")
            modelo.summary(print_fn=logger.info)
            logger.info(f"\nParâmetros de regularização:")
            logger.info(f"- Dimensão do embedding: {embedding_dim}")
            logger.info(f"- Dropout rate: {dropout_rate}")
            logger.info(f"- L2 lambda: {l2_lambda}")
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise


    
    
    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo com os dados fornecidos e técnicas para reduzir overfitting.

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
            
            # Calcular pesos das classes para balanceamento
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            total = n_neg + n_pos
            weight_for_0 = (1 / n_neg) * (total / 2.0)
            weight_for_1 = (1 / n_pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            
            logger.info(f"Pesos das classes: Classe 0: {weight_for_0:.2f}, Classe 1: {weight_for_1:.2f}")
            
            # Construir modelo
            self.modelo = self._construir_modelo_neural(
                len(self.itens_usuario),
                self.item_count
            )
            
            # Configurar callbacks aprimorados
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=4,  # Aumentado para dar mais chances ao modelo
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,  # Redução mais agressiva do learning rate
                    patience=2,
                    min_lr=0.00001,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'modelos/checkpoints/modelo_epoch_{epoch:02d}.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            ]
            
            # Treinar modelo com técnicas anti-overfitting
            logger.info("Iniciando treinamento do modelo neural")
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                validation_split=0.2,
                epochs=15,  # Aumentado número de épocas
                batch_size=32,  # Reduzido batch size
                callbacks=callbacks,
                class_weight=class_weight,  # Adicionado balanceamento de classes
                shuffle=True,  # Garantir shuffle dos dados
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
                    "n_exemplos_positivos": n_pos,
                    "n_exemplos_negativos": n_neg,
                    "ratio_classes": n_pos / n_neg
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
            raise'''

import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization, LayerNormalization
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

class CastLayer(tf.keras.layers.Layer):
    def __init__(self, target_dtype, **kwargs):
        super().__init__(**kwargs)
        self._target_dtype = target_dtype  # Use um nome diferente para o atributo

    def call(self, inputs):
        return tf.cast(inputs, self._target_dtype)

class StringToIndexLayer(tf.keras.layers.Layer):
    def __init__(self, max_tokens, **kwargs):
        super().__init__(**kwargs)
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=1
        )
    
    def adapt(self, data):
        self.vectorize_layer.adapt(data)
    
    def call(self, inputs):
        x = self.vectorize_layer(inputs)
        return tf.cast(x, tf.int32)  # Cast direto aqui, sem usar CastLayer


class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        """
        Inicializa o recomendador híbrido.

        Args:
            dim_embedding: Dimensão dos embeddings para usuários e itens
            dim_features_texto: Dimensão das features de texto
            mlflow_config: Configuração do MLflow
        """
        logger.info("Inicializando RecomendadorHibrido...")
        logger.info(f"Dimensão do embedding: {dim_embedding}")
        logger.info(f"Dimensão das features de texto: {dim_features_texto}")
        
        # Parâmetros do modelo
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        
        # Mapeamentos para itens
        self.item_id_to_index = {}  # Mapeamento de URL para índice
        self.index_to_item_id = {}  # Mapeamento de índice para URL
        self.item_count = 0
        
        # Mapeamentos para usuários
        self.usuario_id_to_index = {}  # Mapeamento de ID do usuário para índice
        self.index_to_usuario_id = {}  # Mapeamento de índice para ID do usuário
        
        # Carregar stopwords
        try:
            stop_words_pt = stopwords.words('portuguese')
            logger.info("Stopwords carregadas do NLTK com sucesso")
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
            logger.info(f"Número de stopwords carregadas: {len(stop_words_pt)}")

        # Inicializar vetorizador TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=dim_features_texto,
            stop_words=stop_words_pt
        )
        
        # Estruturas de dados para armazenar informações dos usuários e itens
        self.itens_usuario = {}  # Histórico de itens por usuário
        self.features_item = {}  # Features dos itens
        
        # Configuração do MLflow
        self.mlflow_config = mlflow_config if mlflow_config else MLflowConfig()
        
        logger.info("RecomendadorHibrido inicializado com sucesso")
        logger.info("Estruturas de dados inicializadas:")
        logger.info(f"- Mapeamentos de itens: {type(self.item_id_to_index)}")
        logger.info(f"- Mapeamentos de usuários: {type(self.usuario_id_to_index)}")
        logger.info(f"- Vetorizador TF-IDF: {type(self.tfidf)}")


    def _limpar_checkpoints(self, caminho_checkpoints="dados/checkpoints"):
        """
        Remove checkpoints antigos para evitar problemas de compatibilidade.
        """
        try:
            caminho = f"{caminho_checkpoints}/exemplos_processados"
            if os.path.exists(caminho):
                logger.info(f"Removendo checkpoint antigo em: {caminho}")
                for arquivo in os.listdir(caminho):
                    os.remove(os.path.join(caminho, arquivo))
                os.rmdir(caminho)
                logger.info("Checkpoint antigo removido com sucesso")
        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")



    def _salvar_checkpoint_exemplos(self, exemplos, caminho_base):
        """
        Salva os exemplos processados em checkpoint.
        """
        try:
            caminho = f"{caminho_base}/exemplos_processados"
            os.makedirs(caminho, exist_ok=True)
            
            # Salvar arrays
            np.save(f"{caminho}/X_usuario.npy", exemplos[0])
            np.save(f"{caminho}/X_item.npy", exemplos[1])
            np.save(f"{caminho}/X_conteudo.npy", exemplos[2])
            np.save(f"{caminho}/y.npy", exemplos[3])
            
            # Converter shapes para listas (serializáveis em JSON)
            metadados = {
                "timestamp": datetime.now().isoformat(),
                "shapes": {
                    "X_usuario": exemplos[0].shape[0] if hasattr(exemplos[0], 'shape') else len(exemplos[0]),
                    "X_item": exemplos[1].shape[0] if hasattr(exemplos[1], 'shape') else len(exemplos[1]),
                    "X_conteudo": list(exemplos[2].shape) if hasattr(exemplos[2], 'shape') else len(exemplos[2]),
                    "y": exemplos[3].shape[0] if hasattr(exemplos[3], 'shape') else len(exemplos[3])
                },
                "n_exemplos": len(exemplos[3]) if hasattr(exemplos[3], '__len__') else 0,
                "dtypes": {
                    "X_usuario": str(exemplos[0].dtype) if hasattr(exemplos[0], 'dtype') else "unknown",
                    "X_item": str(exemplos[1].dtype) if hasattr(exemplos[1], 'dtype') else "unknown",
                    "X_conteudo": str(exemplos[2].dtype) if hasattr(exemplos[2], 'dtype') else "unknown",
                    "y": str(exemplos[3].dtype) if hasattr(exemplos[3], 'dtype') else "unknown"
                }
            }
            
            # Salvar metadados com encoding especificado
            with open(f"{caminho}/metadados.json", "w", encoding='utf-8') as f:
                json.dump(metadados, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Checkpoint salvo em {caminho}")
            logger.info(f"Metadados: {json.dumps(metadados, indent=2)}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            return False

    def _carregar_checkpoint_exemplos(self, caminho_base):
        """
        Carrega exemplos processados de checkpoint.
        """
        try:
            caminho = f"{caminho_base}/exemplos_processados"
            
            if not os.path.exists(caminho):
                logger.info("Checkpoint não encontrado")
                return None
                
            # Verificar se todos os arquivos necessários existem
            arquivos_necessarios = ['X_usuario.npy', 'X_item.npy', 'X_conteudo.npy', 'y.npy', 'metadados.json']
            for arquivo in arquivos_necessarios:
                if not os.path.exists(os.path.join(caminho, arquivo)):
                    logger.warning(f"Arquivo {arquivo} não encontrado no checkpoint")
                    return None
                    
            # Carregar metadados primeiro para verificação
            try:
                with open(f"{caminho}/metadados.json", "r", encoding='utf-8') as f:
                    metadados = json.load(f)
                    logger.info(f"Metadados carregados: {json.dumps(metadados, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar metadados JSON: {str(e)}")
                return None
                
            # Carregar arrays com tratamento de erro
            try:
                X_usuario = np.load(f"{caminho}/X_usuario.npy", allow_pickle=True)
                X_item = np.load(f"{caminho}/X_item.npy")
                X_conteudo = np.load(f"{caminho}/X_conteudo.npy")
                y = np.load(f"{caminho}/y.npy")
            except Exception as e:
                logger.error(f"Erro ao carregar arrays: {str(e)}")
                return None
            
            # Verificar integridade dos dados
            if len(y) != metadados["n_exemplos"]:
                logger.error("Inconsistência entre metadados e dados carregados")
                return None
                
            logger.info(f"Checkpoint carregado de {caminho}")
            logger.info(f"Exemplos carregados: {len(y)}")
            logger.info(f"Shapes: X_usuario {X_usuario.shape}, X_item {X_item.shape}, "
                    f"X_conteudo {X_conteudo.shape}, y {y.shape}")
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {str(e)}")
            return None

    def limpar_checkpoints(self, caminho_checkpoints="dados/checkpoints"):
        """
        Limpa checkpoints antigos.
        """
        try:
            caminho = f"{caminho_checkpoints}/exemplos_processados"
            if os.path.exists(caminho):
                for arquivo in os.listdir(caminho):
                    os.remove(os.path.join(caminho, arquivo))
                os.rmdir(caminho)
                logger.info(f"Checkpoints removidos de {caminho}")
        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")


    def _criar_features_conteudo_pandas(self, dados_itens_pd):
        """
        Cria features de conteúdo usando TF-IDF.
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
        """
        logger.info("Verificando correspondência de URLs")
        
        urls_itens = set(str(page).strip() for page in dados_itens_pd['page'] if page)
        urls_historicos = set()
        for historico in dados_treino_pd['historico']:
            if isinstance(historico, (list, np.ndarray)):
                urls_historicos.update(str(url).strip() for url in historico if url)
        
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
        """
        logger.info("Validando dados de entrada")
        
        if dados_treino_pd.empty:
            raise ValueError("DataFrame de treino está vazio")
        
        colunas_necessarias_treino = ['idUsuario', 'historico']
        colunas_necessarias_itens = ['page', 'conteudo_texto']
        
        for coluna in colunas_necessarias_treino:
            if coluna not in dados_treino_pd.columns:
                raise ValueError(f"Coluna {coluna} não encontrada nos dados de treino")
                
        for coluna in colunas_necessarias_itens:
            if coluna not in dados_itens_pd.columns:
                raise ValueError(f"Coluna {coluna} não encontrada nos dados de itens")
            
        historicos_validos = dados_treino_pd['historico'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
        n_historicos_validos = historicos_validos.sum()
        
        logger.info(f"Total de usuários: {len(dados_treino_pd)}")
        logger.info(f"Usuários com histórico válido: {n_historicos_validos}")
        
        if n_historicos_validos == 0:
            raise ValueError("Nenhum usuário possui histórico válido")
        
        if dados_itens_pd.empty:
            raise ValueError("DataFrame de itens está vazio")
            
        textos_validos = dados_itens_pd['conteudo_texto'].notna()
        n_textos_validos = textos_validos.sum()
        
        logger.info(f"Total de itens: {len(dados_itens_pd)}")
        logger.info(f"Itens com texto válido: {n_textos_validos}")
        
        n_urls_unicas = dados_itens_pd['page'].nunique()
        logger.info(f"URLs únicas nos itens: {n_urls_unicas}")
        
        if 'DataPublicacao' in dados_itens_pd.columns:
            datas_validas = dados_itens_pd['DataPublicacao'].notna()
            n_datas_validas = datas_validas.sum()
            logger.info(f"Itens com datas válidas: {n_datas_validas}")
            
            if n_datas_validas > 0:
                data_min = dados_itens_pd['DataPublicacao'].min()
                data_max = dados_itens_pd['DataPublicacao'].max()
                logger.info(f"Range de datas: {data_min} até {data_max}")
        
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
        """
        logger.info("Iniciando criação de mapeamentos...")
        
        try:
            # Primeiro, mapear usuários únicos
            logger.info("Mapeando usuários para índices...")
            usuarios_unicos = sorted(dados_treino_pd['idUsuario'].unique())
            for idx, usuario_id in enumerate(usuarios_unicos):
                self.usuario_id_to_index[usuario_id] = idx
                self.index_to_usuario_id[idx] = usuario_id
            
            logger.info(f"Total de usuários únicos mapeados: {len(self.usuario_id_to_index)}")
            
            # Depois, mapear itens
            logger.info("Mapeando páginas para índices...")
            dados_itens_pd = dados_itens_pd.reset_index(drop=True)
            
            for idx, page in enumerate(dados_itens_pd['page']):
                if page and isinstance(page, str):
                    page = page.strip()
                    self.item_id_to_index[page] = idx
                    self.index_to_item_id[idx] = page
                    self.item_count = max(self.item_count, idx + 1)
            
            logger.info(f"Total de itens mapeados: {self.item_count}")
            
            # Processar históricos dos usuários
            usuarios_processados = 0
            usuarios_validos = 0
            historicos_invalidos = 0
            
            for _, linha in dados_treino_pd.iterrows():
                try:
                    historico = linha['historico']
                    usuario_id = linha['idUsuario']
                    
                    if historico is not None and isinstance(historico, (list, np.ndarray)):
                        historico_numerico = []
                        for url in historico:
                            url_str = str(url).strip()
                            if url_str in self.item_id_to_index:
                                indice = self.item_id_to_index[url_str]
                                if indice < len(features_conteudo):
                                    historico_numerico.append(indice)
                        
                        if historico_numerico:
                            self.itens_usuario[self.usuario_id_to_index[usuario_id]] = set(historico_numerico)  # Usar índice numérico
                            usuarios_validos += 1
                        else:
                            historicos_invalidos += 1
                    
                    usuarios_processados += 1
                    if usuarios_processados % 1000 == 0:
                        logger.info(f"Processados {usuarios_processados} usuários...")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
            
            # Processar features dos itens
            self.features_item = {}
            for idx in range(self.item_count):
                if idx < len(features_conteudo):
                    self.features_item[idx] = {
                        'vetor_conteudo': features_conteudo[idx],
                        'timestamp': 0
                    }
            
            # Log das estatísticas finais
            logger.info("\n=== ESTATÍSTICAS DE MAPEAMENTO ===")
            logger.info(f"Total de usuários únicos: {len(self.usuario_id_to_index)}")
            logger.info(f"Total de itens únicos: {self.item_count}")
            logger.info(f"Total de usuários processados: {usuarios_processados}")
            logger.info(f"Usuários com histórico válido: {usuarios_validos}")
            logger.info(f"Históricos inválidos: {historicos_invalidos}")
            logger.info(f"Itens com features: {len(self.features_item)}")
            
            # Verificações de integridade
            if usuarios_validos == 0:
                raise ValueError("Nenhum usuário com histórico válido após processamento")
            
            # Exemplo de mapeamento para debug
            primeiro_usuario = list(dados_treino_pd['idUsuario'])[0]
            primeiro_idx = self.usuario_id_to_index[primeiro_usuario]
            logger.info(f"\nExemplo de mapeamento:")
            logger.info(f"ID do primeiro usuário: {primeiro_usuario}")
            logger.info(f"Índice mapeado: {primeiro_idx}")
            logger.info(f"ID recuperado: {self.index_to_usuario_id[primeiro_idx]}")
            
        except Exception as e:
            logger.error(f"Erro durante criação de mapeamentos: {str(e)}")
            raise

    '''def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido mantendo IDs originais dos usuários.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Configurações
            embedding_dim = 16
            dense_units = [32, 16]
            dropout_rate = 0.3
            l2_lambda = 0.01
            
            # Input layers com dtype string para usuários
            entrada_usuario = Input(shape=(1,), dtype=tf.string)
            entrada_item = Input(shape=(1,))
            entrada_conteudo = Input(shape=(self.dim_features_texto,))

            # Camada de hashing para converter strings em índices
            hashing_layer = tf.keras.layers.experimental.preprocessing.Hashing(
                num_bins=n_usuarios,
                salt=123  # valor fixo para consistência
            )
            usuario_hash = hashing_layer(entrada_usuario)

            # Embedding layers
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda)
            )(usuario_hash)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda)
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten()(embedding_usuario)
            item_flat = Flatten()(embedding_item)

            # Concatenação
            concat = Concatenate()([
                usuario_flat,
                item_flat,
                entrada_conteudo
            ])
            
            # Camadas densas
            x = Dense(
                dense_units[0],
                activation='relu',
                kernel_regularizer=l2(l2_lambda)
            )(concat)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            x = Dense(
                dense_units[1],
                activation='relu',
                kernel_regularizer=l2(l2_lambda)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            # Camada de saída
            saida = Dense(1, activation='sigmoid')(x)

            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida
            )
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0
            )
            
            modelo.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            logger.info("\nArquitetura do modelo:")
            modelo.summary(print_fn=logger.info)
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise'''
    
    '''def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido mantendo IDs como strings.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Configurações
            embedding_dim = 16
            dense_units = [32, 16]
            dropout_rate = 0.3
            l2_lambda = 0.01
            
            # Calcular dimensão total da concatenação
            input_dim = (2 * embedding_dim) + self.dim_features_texto  # 2 embeddings + features de texto
            
            # Input layers
            entrada_usuario = Input(shape=(1,), dtype=tf.string, name='usuario_input')
            entrada_item = Input(shape=(1,), dtype=tf.int32, name='item_input')
            entrada_conteudo = Input(shape=(self.dim_features_texto,), name='conteudo_input')

            # Camada de processamento de strings
            string_to_index = StringToIndexLayer(
                max_tokens=n_usuarios + 1, 
                name='string_to_index'
            )
            # Adaptar a camada com todos os IDs de usuários possíveis
            string_to_index.adapt(tf.constant(list(self.itens_usuario.keys())))
            
            # Converter strings para índices
            usuario_idx = string_to_index(entrada_usuario)

            # Embedding layers
            embedding_usuario = Embedding(
                input_dim=n_usuarios + 1,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='usuario_embedding'
            )(usuario_idx)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='item_embedding'
            )(entrada_item)

            # Flatten embeddings
            usuario_flat = Flatten(name='usuario_flatten')(embedding_usuario)
            item_flat = Flatten(name='item_flatten')(embedding_item)

            # Concatenação
            concat = Concatenate(name='concat_layer')([
                usuario_flat,
                item_flat,
                entrada_conteudo
            ])
            
            # Primeira camada densa com input_shape explícito
            x = Dense(
                dense_units[0],
                activation='relu',
                kernel_regularizer=l2(l2_lambda),
                input_shape=(input_dim,),  # Dimensão de entrada explícita
                name='dense_1'
            )(concat)
            x = BatchNormalization(name='batch_norm_1')(x)
            x = Dropout(dropout_rate, name='dropout_1')(x)
            
            # Segunda camada densa
            x = Dense(
                dense_units[1],
                activation='relu',
                kernel_regularizer=l2(l2_lambda),
                name='dense_2'
            )(x)
            x = BatchNormalization(name='batch_norm_2')(x)
            x = Dropout(dropout_rate, name='dropout_2')(x)

            # Camada de saída
            saida = Dense(1, activation='sigmoid', name='output')(x)

            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=saida,
                name='modelo_recomendacao'
            )
            
            # Compilar modelo
            modelo.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Log das dimensões
            logger.info("\nDimensões das camadas:")
            logger.info(f"Input dim total: {input_dim}")
            logger.info(f"Embedding dim: {embedding_dim}")
            logger.info(f"Features texto dim: {self.dim_features_texto}")
            
            logger.info("\nArquitetura do modelo:")
            modelo.summary(print_fn=logger.info)
            
            return modelo
                
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise'''

    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural híbrido usando índices numéricos para usuários.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Configurações
            embedding_dim = 16
            dense_units = [32, 16]
            dropout_rate = 0.3
            l2_lambda = 0.01
            
            # Calcular dimensão de entrada fixa
            input_dim = (2 * embedding_dim) + self.dim_features_texto
            logger.info(f"Dimensão de entrada total: {input_dim}")
            
            # Definir as camadas de entrada
            entrada_usuario = Input(shape=(1,), dtype=tf.int32, name='usuario_input')
            entrada_item = Input(shape=(1,), dtype=tf.int32, name='item_input')
            entrada_conteudo = Input(shape=(self.dim_features_texto,), dtype=tf.float32, name='conteudo_input')
            
            # Embeddings para usuários
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='usuario_embedding'
            )(entrada_usuario)
            
            # Embeddings para itens
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='item_embedding'
            )(entrada_item)
            
            # Flatten dos embeddings
            usuario_flat = Flatten(name='usuario_flatten')(embedding_usuario)
            item_flat = Flatten(name='item_flatten')(embedding_item)
            
            # Concatenar todas as features
            concat = Concatenate(name='concat_layer')([
                usuario_flat,
                item_flat,
                entrada_conteudo
            ])
            
            # Camadas densas
            x = Dense(
                units=dense_units[0],
                activation='relu',
                kernel_regularizer=l2(l2_lambda),
                name='dense_1'
            )(concat)
            x = LayerNormalization(name='norm_1')(x)
            x = Dropout(dropout_rate, name='dropout_1')(x)
            
            x = Dense(
                units=dense_units[1],
                activation='relu',
                kernel_regularizer=l2(l2_lambda),
                name='dense_2'
            )(x)
            x = LayerNormalization(name='norm_2')(x)
            x = Dropout(dropout_rate, name='dropout_2')(x)
            
            # Camada de saída
            output = Dense(1, activation='sigmoid', name='output')(x)
            
            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=output,
                name='modelo_recomendacao'
            )
            
            # Compilar modelo
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0,
                epsilon=1e-7
            )
            
            modelo.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Log das dimensões
            logger.info("\nDimensões do modelo:")
            logger.info(f"Embedding dim: {embedding_dim}")
            logger.info(f"Input dim total: {input_dim}")
            logger.info(f"Dense units: {dense_units}")
            logger.info(f"Número de usuários: {n_usuarios}")
            logger.info(f"Número de itens: {n_itens}")
            logger.info(f"Dimensão features texto: {self.dim_features_texto}")
            
            modelo.summary(print_fn=logger.info)
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
            raise

    '''def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, 
                                    max_exemplos_total=1000000,
                                    max_exemplos_por_usuario=100,
                                    batch_size=10000):
        """
        Prepara os dados de treino mantendo IDs originais dos usuários.
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
            
            # Processar exemplos
            for usuario_id, historico in self.itens_usuario.items():
                # Limitar exemplos por usuário
                n_exemplos_usuario = min(len(historico), max_exemplos_por_usuario // 2)
                
                # Processar exemplos positivos
                for item_idx in list(historico)[:n_exemplos_usuario]:
                    if item_idx in self.features_item:
                        X_usuario_list.append(str(usuario_id))  # Converter para string
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(1)
                
                # Processar exemplos negativos
                todos_itens = set(self.features_item.keys())
                itens_negativos = todos_itens - historico
                for item_idx in list(itens_negativos)[:n_exemplos_usuario]:
                    X_usuario_list.append(str(usuario_id))  # Converter para string
                    X_item_list.append(item_idx)
                    X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                    y_list.append(0)
                
                if len(y_list) >= max_exemplos_total:
                    break
            
            # Converter para arrays
            X_usuario = np.array(X_usuario_list)
            X_item = np.array(X_item_list, dtype=np.int32)
            X_conteudo = np.array(X_conteudo_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
            logger.info("\n=== ESTATÍSTICAS FINAIS ===")
            logger.info(f"Total de exemplos: {len(y)}")
            logger.info(f"Exemplos positivos: {np.sum(y == 1)}")
            logger.info(f"Exemplos negativos: {np.sum(y == 0)}")
            logger.info(f"Forma dos arrays:")
            logger.info(f"X_usuario: {X_usuario.shape}, dtype: {X_usuario.dtype}")
            logger.info(f"X_item: {X_item.shape}, dtype: {X_item.dtype}")
            logger.info(f"X_conteudo: {X_conteudo.shape}, dtype: {X_conteudo.dtype}")
            logger.info(f"y: {y.shape}, dtype: {y.dtype}")
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise'''
    

    def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, 
                                   max_exemplos_total=1000000,
                                   max_exemplos_por_usuario=100,
                                   batch_size=10000,
                                   caminho_checkpoints="dados/checkpoints"):
        """
        Prepara os dados de treino usando índices numéricos para usuários.
        """
        logger.info("Preparando dados de treino em lotes")
        
        # Tentar carregar checkpoint
        exemplos = self._carregar_checkpoint_exemplos(caminho_checkpoints)
        if exemplos is not None:
            return exemplos
            
        logger.info("Checkpoint não encontrado, processando dados...")
        
        X_usuario_list = []
        X_item_list = []
        X_conteudo_list = []
        y_list = []
        
        try:
            # Converter itens_usuario para usar strings como chaves
            itens_usuario_str = {str(k): v for k, v in self.itens_usuario.items()}
            self.itens_usuario = itens_usuario_str
            
            # Criar mapeamento de usuários
            logger.info("Criando mapeamento de usuários...")
            
            # Converter todos os IDs para string
            usuarios_df = set(str(uid) for uid in dados_treino_pd['idUsuario'].unique())
            usuarios_hist = set(self.itens_usuario.keys())  # Já está em string
            usuarios_unicos = sorted(usuarios_df | usuarios_hist)
            
            # Criar mapeamentos
            self.usuario_id_to_index = {usuario_id: idx for idx, usuario_id in enumerate(usuarios_unicos)}
            self.index_to_usuario_id = {idx: usuario_id for usuario_id, idx in self.usuario_id_to_index.items()}
            
            logger.info(f"Mapeamento criado para {len(self.usuario_id_to_index)} usuários")
            
            # Debug do mapeamento
            logger.info("\n=== DEBUG DO MAPEAMENTO ===")
            logger.info(f"Total de usuários mapeados: {len(self.usuario_id_to_index)}")
            logger.info(f"Total de usuários no histórico: {len(self.itens_usuario)}")
            logger.info(f"Exemplo de IDs: {list(self.usuario_id_to_index.keys())[:5]}")
            
            # Verificar primeiro usuário
            if self.itens_usuario:
                primeiro_usuario = list(self.itens_usuario.keys())[0]
                logger.info(f"Tentando acessar primeiro usuário: {primeiro_usuario}")
                logger.info(f"Tipos de chaves em itens_usuario: {[type(k) for k in list(self.itens_usuario.keys())[:5]]}")
                
                if primeiro_usuario in self.usuario_id_to_index:
                    primeiro_usuario_idx = self.usuario_id_to_index[primeiro_usuario]
                    historico_primeiro_usuario = self.itens_usuario[primeiro_usuario]
                    
                    logger.info("\n=== EXEMPLOS DETALHADOS DO PRIMEIRO USUÁRIO ===")
                    logger.info(f"ID do Usuário: {primeiro_usuario} (tipo: {type(primeiro_usuario)})")
                    logger.info(f"Índice do usuário: {primeiro_usuario_idx}")
                    logger.info(f"Tamanho do histórico: {len(historico_primeiro_usuario)}")
                else:
                    logger.warning(f"Primeiro usuário {primeiro_usuario} não encontrado no mapeamento")
            
            # Lista de todos os itens disponíveis
            todos_itens = list(self.features_item.keys())
            
            # Processar usuários em lotes
            usuarios_processados = 0
            usuarios_com_erro = 0
            total_usuarios = len(self.itens_usuario)
            
            for usuario_id, historico in self.itens_usuario.items():
                try:
                    # Verificar limite total de exemplos
                    if len(y_list) >= max_exemplos_total:
                        logger.info(f"Atingido limite máximo de exemplos: {max_exemplos_total}")
                        break
                    
                    # ID já está em string
                    if usuario_id not in self.usuario_id_to_index:
                        logger.warning(f"Usuário {usuario_id} não encontrado no mapeamento")
                        usuarios_com_erro += 1
                        continue
                    
                    usuario_idx = self.usuario_id_to_index[usuario_id]
                    
                    # Limitar exemplos positivos por usuário
                    n_exemplos_positivos = min(len(historico), max_exemplos_por_usuario // 2)
                    historico_amostrado = list(historico)
                    if len(historico_amostrado) > n_exemplos_positivos:
                        historico_amostrado = np.random.choice(
                            historico_amostrado, 
                            n_exemplos_positivos, 
                            replace=False
                        )
                    
                    # Adicionar exemplos positivos
                    for item_idx in historico_amostrado:
                        if item_idx in self.features_item:
                            X_usuario_list.append(usuario_idx)
                            X_item_list.append(item_idx)
                            X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                            y_list.append(1)
                    
                    # Adicionar exemplos negativos
                    n_exemplos_negativos = len(historico_amostrado)
                    itens_negativos = np.random.choice(
                        [i for i in todos_itens if i not in historico],
                        size=min(n_exemplos_negativos, max_exemplos_por_usuario // 2),
                        replace=False
                    )
                    
                    for item_idx in itens_negativos:
                        X_usuario_list.append(usuario_idx)
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(0)
                    
                    # Log de progresso
                    usuarios_processados += 1
                    if usuarios_processados % 100 == 0:
                        logger.info(f"Processados {usuarios_processados}/{total_usuarios} usuários "
                                f"({len(y_list)} exemplos gerados)")
                        
                        # Debug periódico
                        if len(X_usuario_list) > 0:
                            logger.info(f"Último índice de usuário: {X_usuario_list[-1]}")
                            logger.info(f"ID original: {self.index_to_usuario_id[X_usuario_list[-1]]}")
                
                except Exception as e:
                    logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
                    usuarios_com_erro += 1
                    continue
            
            # Verificar se temos dados suficientes
            if len(X_usuario_list) == 0:
                raise ValueError("Nenhum exemplo de treino gerado")
            
            # Converter para arrays numpy
            X_usuario = np.array(X_usuario_list, dtype=np.int32)
            X_item = np.array(X_item_list, dtype=np.int32)
            X_conteudo = np.array(X_conteudo_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            
            # Log final
            logger.info("\n=== ESTATÍSTICAS FINAIS ===")
            logger.info(f"Total de exemplos: {len(y)}")
            logger.info(f"Exemplos positivos: {np.sum(y == 1)}")
            logger.info(f"Exemplos negativos: {np.sum(y == 0)}")
            logger.info(f"Usuários processados com sucesso: {usuarios_processados}")
            logger.info(f"Usuários com erro: {usuarios_com_erro}")
            logger.info(f"Shape X_usuario: {X_usuario.shape}, dtype: {X_usuario.dtype}")
            logger.info(f"Range de índices: [{np.min(X_usuario)}, {np.max(X_usuario)}]")
            
            # Verificações adicionais
            logger.info("\n=== VERIFICAÇÕES DE INTEGRIDADE ===")
            usuarios_unicos = np.unique(X_usuario)
            logger.info(f"Número de usuários únicos nos dados: {len(usuarios_unicos)}")
            logger.info(f"Primeiros 5 índices de usuários: {usuarios_unicos[:5]}")
            logger.info(f"IDs originais correspondentes: {[self.index_to_usuario_id[idx] for idx in usuarios_unicos[:5]]}")
            
            # Salvar checkpoint
            exemplos = (X_usuario, X_item, X_conteudo, y)
            self._salvar_checkpoint_exemplos(exemplos, caminho_checkpoints)
            
            return exemplos
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise

    def treinar(self, dados_treino, dados_itens):
        """
        Treina o modelo com os dados fornecidos.
        """
        logger.info("Iniciando treinamento do modelo")
        
        # Configurações de memória
        try:
            tf.config.set_soft_device_placement(True)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception as e:
            logger.warning(f"Erro ao configurar memória TF: {str(e)}")
        
        try:
            # Verificar se existem exemplos processados salvos
            caminho_checkpoints = "dados/checkpoints"
            caminho_exemplos = f"{caminho_checkpoints}/exemplos_processados"
            
            if os.path.exists(f"{caminho_exemplos}/X_usuario.npy"):
                logger.info("Carregando dados processados do checkpoint...")
                try:
                    X_usuario = np.load(f"{caminho_exemplos}/X_usuario.npy")
                    X_item = np.load(f"{caminho_exemplos}/X_item.npy")
                    X_conteudo = np.load(f"{caminho_exemplos}/X_conteudo.npy")
                    y = np.load(f"{caminho_exemplos}/y.npy")
                    
                    logger.info("Dados carregados com sucesso do checkpoint!")
                    logger.info(f"Dimensões dos dados carregados:")
                    logger.info(f"X_usuario: {X_usuario.shape}")
                    logger.info(f"X_item: {X_item.shape}")
                    logger.info(f"X_conteudo: {X_conteudo.shape}")
                    logger.info(f"y: {y.shape}")
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar checkpoint: {str(e)}")
                    logger.info("Processando dados novamente...")
                    # Processar dados normalmente
                    X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                        dados_treino_pd, 
                        features_conteudo,
                        caminho_checkpoints=caminho_checkpoints
                    )
            else:
                logger.info("Checkpoint não encontrado. Criando exemplos...")
            
            # Filtrar datas válidas
            dados_itens = dados_itens.filter(
                (F.year(F.col("DataPublicacao")) >= 1970) &
                (F.year(F.col("DataPublicacao")) <= 2030)
            )
            
            # Converter dados Spark para pandas
            logger.info("Convertendo dados Spark para pandas")
            dados_treino_pd = dados_treino.toPandas()
            dados_itens_pd = dados_itens.toPandas()
            
            # Validar dados
            self._validar_dados_entrada(dados_treino_pd, dados_itens_pd)
            self._verificar_correspondencia_ids(dados_treino_pd, dados_itens_pd)
            
            # Criar features de conteúdo
            features_conteudo = self._criar_features_conteudo_pandas(dados_itens_pd)
            
            # Criar mapeamentos
            self._criar_mapeamentos(dados_treino_pd, dados_itens_pd, features_conteudo)
            
            # Preparar dados de treino
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd, 
                features_conteudo,
                caminho_checkpoints="dados/checkpoints"
            )
            
            if len(X_usuario) == 0:
                raise ValueError("Nenhum exemplo de treino gerado")
                
            # Verificar tipos de dados antes da conversão
            logger.info("\nTipos de dados antes da conversão:")
            logger.info(f"X_usuario dtype: {X_usuario.dtype}")
            logger.info(f"Amostra X_usuario: {X_usuario[:5]}")
            
            # Garantir que X_usuario seja int32
            if not np.issubdtype(X_usuario.dtype, np.integer):
                logger.warning("Convertendo X_usuario para int32")
                X_usuario = X_usuario.astype(np.int32)
            
            # Converter para tensores TensorFlow com tipos corretos
            X_usuario = tf.convert_to_tensor(X_usuario, dtype=tf.int32)
            X_item = tf.convert_to_tensor(X_item, dtype=tf.int32)
            X_conteudo = tf.convert_to_tensor(X_conteudo, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            
            # Log das dimensões e tipos
            logger.info("\nDimensões e tipos dos dados de treino:")
            logger.info(f"X_usuario: shape={X_usuario.shape}, dtype={X_usuario.dtype}")
            logger.info(f"X_item: shape={X_item.shape}, dtype={X_item.dtype}")
            logger.info(f"X_conteudo: shape={X_conteudo.shape}, dtype={X_conteudo.dtype}")
            logger.info(f"y: shape={y.shape}, dtype={y.dtype}")
            
            # Verificações adicionais
            logger.info("\nVerificações de integridade:")
            logger.info(f"Valor máximo X_usuario: {tf.reduce_max(X_usuario)}")
            logger.info(f"Valor máximo X_item: {tf.reduce_max(X_item)}")
            logger.info(f"Número de usuários únicos: {len(tf.unique(X_usuario)[0])}")
            logger.info(f"Número de itens únicos: {len(tf.unique(X_item)[0])}")
            
            # Construir modelo
            self.modelo = self._construir_modelo_neural(
                len(self.usuario_id_to_index),  # Número de usuários únicos
                self.item_count
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=4,
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=0.00001,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='dados/checkpoints/modelo_{epoch:02d}_{val_loss:.2f}.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    save_weights_only=False
                )
            ]
            
            # Treinar modelo
            logger.info("Iniciando treinamento...")
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                validation_split=0.2,
                epochs=15,
                batch_size=16,  # Reduzido para economizar memória
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            
            # Log das métricas finais
            logger.info("\nMétricas finais:")
            for metrica, valores in historia.history.items():
                logger.info(f"{metrica}: {valores[-1]:.4f}")
            
            # Registrar métricas no MLflow
            if mlflow.active_run():
                metricas = {
                    "loss_final": historia.history['loss'][-1],
                    "val_loss_final": historia.history['val_loss'][-1],
                    "accuracy_final": historia.history['accuracy'][-1],
                    "val_accuracy_final": historia.history['val_accuracy'][-1],
                    "n_exemplos_treino": len(y),
                    "n_exemplos_positivos": int(tf.reduce_sum(tf.cast(y == 1, tf.int32))),
                    "n_exemplos_negativos": int(tf.reduce_sum(tf.cast(y == 0, tf.int32))),
                    "n_usuarios": len(self.usuario_id_to_index),
                    "n_itens": self.item_count
                }
                self.mlflow_config.log_metricas(metricas)
            
            return historia
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise e

    def prever(self, usuario_id, candidatos=None, k=10):
        """
        Faz previsões para um usuário mantendo ID original.
        """
        if not self.modelo:
            raise ValueError("Modelo não treinado")
            
        if usuario_id not in self.itens_usuario:
            logger.warning(f"Usuário {usuario_id} não encontrado no conjunto de treino")
            return []
        
        if not candidatos:
            candidatos = list(self.features_item.keys())
        
        # Preparar dados para previsão
        X_usuario = np.array([str(usuario_id)] * len(candidatos))
        X_item = np.array(candidatos, dtype=np.int32)
        X_conteudo = np.array([self.features_item[idx]['vetor_conteudo'] for idx in candidatos], dtype=np.float32)
        
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
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise

    @classmethod
    def carregar_modelo(cls, caminho):
        """
        Carrega um modelo salvo do disco.
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
            
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise