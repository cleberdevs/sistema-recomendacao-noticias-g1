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
        self._target_dtype = target_dtype

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
        return tf.cast(x, tf.int32)

class RecomendadorHibrido:
    def __init__(self, dim_embedding=32, dim_features_texto=100, mlflow_config=None):
        logger.info("Inicializando RecomendadorHibrido...")
        logger.info(f"Dimensão do embedding: {dim_embedding}")
        logger.info(f"Dimensão das features de texto: {dim_features_texto}")
        
        self.dim_embedding = dim_embedding
        self.dim_features_texto = dim_features_texto
        self.modelo = None
        self.item_id_to_index = {}
        self.index_to_item_id = {}
        self.item_count = 0
        self.usuario_id_to_index = {}
        self.index_to_usuario_id = {}
        
        try:
            stop_words_pt = stopwords.words('portuguese')
            logger.info("Stopwords carregadas do NLTK com sucesso")
        except:
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
        
        logger.info("RecomendadorHibrido inicializado com sucesso")


    def _validar_dados_entrada(self, dados_treino_pd, dados_itens_pd):
        """
        Valida os dados de entrada antes do processamento.
        """
        logger.info("Validando dados de entrada")
        
        try:
            if dados_treino_pd.empty:
                raise ValueError("DataFrame de treino está vazio")
            
            if dados_itens_pd.empty:
                raise ValueError("DataFrame de itens está vazio")
            
            # Verificar colunas necessárias
            colunas_necessarias_treino = ['idUsuario', 'historico']
            colunas_necessarias_itens = ['page', 'conteudo_texto']
            
            colunas_faltantes_treino = [col for col in colunas_necessarias_treino if col not in dados_treino_pd.columns]
            if colunas_faltantes_treino:
                raise ValueError(f"Colunas faltantes nos dados de treino: {colunas_faltantes_treino}")
            
            colunas_faltantes_itens = [col for col in colunas_necessarias_itens if col not in dados_itens_pd.columns]
            if colunas_faltantes_itens:
                raise ValueError(f"Colunas faltantes nos dados de itens: {colunas_faltantes_itens}")
            
            # Verificar valores nulos
            nulos_treino = dados_treino_pd[colunas_necessarias_treino].isnull().sum()
            if nulos_treino.any():
                logger.warning(f"Valores nulos encontrados nos dados de treino:\n{nulos_treino}")
            
            nulos_itens = dados_itens_pd[colunas_necessarias_itens].isnull().sum()
            if nulos_itens.any():
                logger.warning(f"Valores nulos encontrados nos dados de itens:\n{nulos_itens}")
            
            # Verificar históricos
            historicos_validos = dados_treino_pd['historico'].apply(
                lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
            )
            n_historicos_validos = historicos_validos.sum()
            
            logger.info(f"Total de usuários: {len(dados_treino_pd)}")
            logger.info(f"Usuários com histórico válido: {n_historicos_validos}")
            
            if n_historicos_validos == 0:
                raise ValueError("Nenhum usuário possui histórico válido")
            
            # Verificar textos dos itens
            textos_validos = dados_itens_pd['conteudo_texto'].notna()
            n_textos_validos = textos_validos.sum()
            
            logger.info(f"Total de itens: {len(dados_itens_pd)}")
            logger.info(f"Itens com texto válido: {n_textos_validos}")
            
            # Verificar URLs únicas
            n_urls_unicas = dados_itens_pd['page'].nunique()
            logger.info(f"URLs únicas nos itens: {n_urls_unicas}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação dos dados: {str(e)}")
            raise

    def _verificar_correspondencia_ids(self, dados_treino_pd, dados_itens_pd):
        """
        Verifica a correspondência entre URLs nos históricos e no DataFrame de itens.
        """
        logger.info("Verificando correspondência de URLs")
        
        try:
            # Coletar URLs únicas dos itens
            urls_itens = set(str(page).strip() for page in dados_itens_pd['page'] if pd.notna(page))
            
            # Coletar URLs dos históricos
            urls_historicos = set()
            for historico in dados_treino_pd['historico']:
                if isinstance(historico, (list, np.ndarray)):
                    urls_historicos.update(str(url).strip() for url in historico if pd.notna(url))
            
            # Análise de sobreposição
            urls_comuns = urls_itens.intersection(urls_historicos)
            urls_apenas_historico = urls_historicos - urls_itens
            urls_apenas_itens = urls_itens - urls_historicos
            
            # Log das estatísticas
            logger.info(f"URLs nos itens: {len(urls_itens)}")
            logger.info(f"URLs nos históricos: {len(urls_historicos)}")
            logger.info(f"URLs comuns: {len(urls_comuns)}")
            logger.info(f"URLs apenas no histórico: {len(urls_apenas_historico)}")
            logger.info(f"URLs apenas nos itens: {len(urls_apenas_itens)}")
            
            # Avisos sobre URLs não encontradas
            if urls_apenas_historico:
                logger.warning("Exemplos de URLs no histórico mas não nos itens:")
                logger.warning(list(urls_apenas_historico)[:5])
            
            # Verificar cobertura mínima
            cobertura = len(urls_comuns) / len(urls_historicos) if urls_historicos else 0
            logger.info(f"Cobertura de URLs: {cobertura:.2%}")
            
            if cobertura < 0.5:  # 50% de cobertura mínima
                logger.warning("Baixa cobertura de URLs entre históricos e itens")
            
            return urls_comuns, urls_apenas_historico, urls_apenas_itens
            
        except Exception as e:
            logger.error(f"Erro ao verificar correspondência de IDs: {str(e)}")
            raise

    def _criar_features_conteudo_pandas(self, dados_itens_pd):
        """
        Cria features de conteúdo usando TF-IDF.
        """
        logger.info("Criando features de conteúdo")
        
        try:
            # Preparar textos
            textos = dados_itens_pd['conteudo_texto'].fillna('')
            
            # Verificar dados
            n_textos_vazios = (textos == '').sum()
            if n_textos_vazios > 0:
                logger.warning(f"Encontrados {n_textos_vazios} textos vazios")
            
            # Criar features
            features = self.tfidf.fit_transform(textos).toarray()
            
            logger.info(f"Features de conteúdo criadas com forma: {features.shape}")
            logger.info(f"Número de termos no vocabulário: {len(self.tfidf.vocabulary_)}")
            
            # Verificar qualidade das features
            media_features = np.mean(features, axis=1)
            n_zeros = np.sum(media_features == 0)
            if n_zeros > 0:
                logger.warning(f"{n_zeros} itens com todas as features zeradas")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao criar features de conteúdo: {str(e)}")
            raise    

    def _criar_mapeamentos(self, dados_treino_pd, dados_itens_pd, features_conteudo):
        """
        Cria mapeamentos entre IDs e índices numéricos para itens e usuários.
        """
        logger.info("Iniciando criação de mapeamentos")
        
        try:
            # Mapear usuários únicos
            logger.info("Mapeando usuários para índices...")
            usuarios_unicos = sorted(set(str(uid) for uid in dados_treino_pd['idUsuario'].unique()))
            self.usuario_id_to_index = {str(uid): idx for idx, uid in enumerate(usuarios_unicos)}
            self.index_to_usuario_id = {idx: uid for uid, idx in self.usuario_id_to_index.items()}
            
            logger.info(f"Total de usuários únicos mapeados: {len(self.usuario_id_to_index)}")
            
            # Mapear itens
            logger.info("Mapeando páginas para índices...")
            dados_itens_pd = dados_itens_pd.reset_index(drop=True)
            
            self.item_count = 0
            for idx, page in enumerate(dados_itens_pd['page']):
                if page and isinstance(page, str):
                    page = page.strip()
                    self.item_id_to_index[page] = idx
                    self.index_to_item_id[idx] = page
                    self.item_count = max(self.item_count, idx + 1)
            
            logger.info(f"Total de itens mapeados: {self.item_count}")
            
            # Processar históricos dos usuários
            logger.info("Processando históricos dos usuários...")
            self.itens_usuario = {}
            usuarios_processados = 0
            usuarios_validos = 0
            historicos_invalidos = 0
            
            for _, linha in dados_treino_pd.iterrows():
                try:
                    historico = linha['historico']
                    usuario_id = str(linha['idUsuario'])
                    
                    if historico is not None and isinstance(historico, (list, np.ndarray)):
                        historico_numerico = []
                        for url in historico:
                            url_str = str(url).strip()
                            if url_str in self.item_id_to_index:
                                indice = self.item_id_to_index[url_str]
                                if indice < len(features_conteudo):
                                    historico_numerico.append(indice)
                        
                        if historico_numerico:
                            self.itens_usuario[usuario_id] = set(historico_numerico)
                            usuarios_validos += 1
                        else:
                            historicos_invalidos += 1
                    
                    usuarios_processados += 1
                    if usuarios_processados % 1000 == 0:
                        logger.info(f"Processados {usuarios_processados} usuários...")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
                    continue
            
            # Processar features dos itens
            logger.info("Processando features dos itens...")
            self.features_item = {}
            for idx in range(self.item_count):
                if idx < len(features_conteudo):
                    self.features_item[idx] = {
                        'vetor_conteudo': features_conteudo[idx],
                        'timestamp': 0  # Placeholder para timestamp
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
            if self.usuario_id_to_index:
                primeiro_usuario = list(self.usuario_id_to_index.keys())[0]
                primeiro_idx = self.usuario_id_to_index[primeiro_usuario]
                logger.info(f"\nExemplo de mapeamento:")
                logger.info(f"ID do primeiro usuário: {primeiro_usuario}")
                logger.info(f"Índice mapeado: {primeiro_idx}")
                logger.info(f"ID recuperado: {self.index_to_usuario_id[primeiro_idx]}")
            
        except Exception as e:
            logger.error(f"Erro durante criação de mapeamentos: {str(e)}")
            raise        

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


    def _construir_modelo_neural(self, n_usuarios, n_itens):
        """
        Constrói o modelo neural com configurações otimizadas.
        """
        logger.info(f"Construindo modelo neural com {n_usuarios} usuários e {n_itens} itens")
        
        try:
            # Adicionar margem de segurança
            n_usuarios = int(n_usuarios * 1.1)  # 10% de margem
            n_itens = int(n_itens * 1.1)  # 10% de margem
            
            # Configurações otimizadas
            embedding_dim = 16
            dense_units = [32, 16]
            dropout_rate = 0.3
            l2_lambda = 0.01
            
            # Dimensão de entrada
            input_dim = (2 * embedding_dim) + self.dim_features_texto
            
            # Camadas de entrada
            entrada_usuario = Input(shape=(1,), dtype=tf.int32, name='usuario_input')
            entrada_item = Input(shape=(1,), dtype=tf.int32, name='item_input')
            entrada_conteudo = Input(shape=(self.dim_features_texto,), dtype=tf.float32, name='conteudo_input')
            
            # Embeddings com margem de segurança
            embedding_usuario = Embedding(
                input_dim=n_usuarios,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='usuario_embedding'
            )(entrada_usuario)
            
            embedding_item = Embedding(
                input_dim=n_itens,
                output_dim=embedding_dim,
                embeddings_regularizer=l2(l2_lambda),
                name='item_embedding'
            )(entrada_item)
            
            # Flatten
            usuario_flat = Flatten(name='usuario_flatten')(embedding_usuario)
            item_flat = Flatten(name='item_flatten')(embedding_item)
            
            # Concatenação
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
            
            # Saída
            output = Dense(1, activation='sigmoid', name='output')(x)
            
            # Criar modelo
            modelo = Model(
                inputs=[entrada_usuario, entrada_item, entrada_conteudo],
                outputs=output,
                name='modelo_recomendacao'
            )
            
            # Compilar
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
            logger.info(f"Número de usuários (com margem): {n_usuarios}")
            logger.info(f"Número de itens (com margem): {n_itens}")
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo neural: {str(e)}")
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
            # Limitar uso de memória GPU se disponível
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Erro ao configurar memória TF: {str(e)}")
        
        try:
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
            
            # Preparar dados de treino com limites reduzidos
            X_usuario, X_item, X_conteudo, y = self._preparar_dados_treino_em_lotes(
                dados_treino_pd, 
                features_conteudo,
                max_exemplos_total=500000,  # Reduzido
                max_exemplos_por_usuario=50,  # Reduzido
                caminho_checkpoints="dados/checkpoints"
            )
            
            if len(X_usuario) == 0:
                raise ValueError("Nenhum exemplo de treino gerado")
            
            # Garantir tipos corretos
            X_usuario = tf.convert_to_tensor(X_usuario.astype(np.int32), dtype=tf.int32)
            X_item = tf.convert_to_tensor(X_item.astype(np.int32), dtype=tf.int32)
            X_conteudo = tf.convert_to_tensor(X_conteudo.astype(np.float32), dtype=tf.float32)
            y = tf.convert_to_tensor(y.astype(np.float32), dtype=tf.float32)
            
            # Verificações de integridade
            max_usuario_idx = tf.reduce_max(X_usuario)
            max_item_idx = tf.reduce_max(X_item)
            
            if max_usuario_idx >= len(self.usuario_id_to_index):
                raise ValueError(f"Índice de usuário {max_usuario_idx} maior que o número de usuários {len(self.usuario_id_to_index)}")
            if max_item_idx >= self.item_count:
                raise ValueError(f"Índice de item {max_item_idx} maior que o número de itens {self.item_count}")
            
            # Construir modelo com margem de segurança
            self.modelo = self._construir_modelo_neural(
                int(len(self.usuario_id_to_index) * 1.1),  # 10% de margem
                int(self.item_count * 1.1)  # 10% de margem
            )
            
            # Callbacks otimizados
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=2,  # Reduzido
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=1,  # Reduzido
                    min_lr=0.0001,
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
            
            # Treinar com parâmetros otimizados
            logger.info("Iniciando treinamento...")
            historia = self.modelo.fit(
                [X_usuario, X_item, X_conteudo],
                y,
                validation_split=0.2,
                epochs=5,  # Reduzido
                batch_size=512,  # Aumentado
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            
            # Log e registro de métricas
            logger.info("\nMétricas finais:")
            for metrica, valores in historia.history.items():
                logger.info(f"{metrica}: {valores[-1]:.4f}")
            
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
                    "n_itens": self.item_count,
                    "max_usuario_idx": int(max_usuario_idx),
                    "max_item_idx": int(max_item_idx)
                }
                self.mlflow_config.log_metricas(metricas)
            
            return historia
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            raise e

    def _preparar_dados_treino_em_lotes(self, dados_treino_pd, features_conteudo, 
                                       max_exemplos_total=500000,
                                       max_exemplos_por_usuario=50,
                                       batch_size=1000,
                                       caminho_checkpoints="dados/checkpoints"):
        """
        Prepara os dados de treino com melhor tratamento de erros.
        """
        logger.info("Preparando dados de treino em lotes")
        
        # Tentar carregar checkpoint
        exemplos = self._carregar_checkpoint_exemplos(caminho_checkpoints)
        if exemplos is not None:
            return exemplos
        
        X_usuario_list = []
        X_item_list = []
        X_conteudo_list = []
        y_list = []
        
        try:
            usuarios_processados = 0
            usuarios_ignorados = 0
            total_exemplos = 0
            
            # Converter todos os IDs para string para garantir consistência
            itens_usuario_processados = {
                str(k): v for k, v in self.itens_usuario.items()
            }
            
            # Log inicial
            logger.info(f"Total de usuários para processar: {len(itens_usuario_processados)}")
            logger.info(f"Total de usuários mapeados: {len(self.usuario_id_to_index)}")
            
            # Processar usuários em lotes
            for usuario_id, historico in itens_usuario_processados.items():
                try:
                    if total_exemplos >= max_exemplos_total:
                        logger.info(f"Atingido limite máximo de exemplos: {max_exemplos_total}")
                        break
                    
                    if str(usuario_id) not in self.usuario_id_to_index:
                        usuarios_ignorados += 1
                        if usuarios_ignorados % 100 == 0:
                            logger.warning(f"Usuário {usuario_id} não encontrado no mapeamento. "
                                         f"Total ignorados: {usuarios_ignorados}")
                        continue
                    
                    usuario_idx = self.usuario_id_to_index[str(usuario_id)]
                    
                    # Validar histórico
                    historico_valido = [
                        item_idx for item_idx in historico 
                        if item_idx in self.features_item
                    ]
                    
                    if not historico_valido:
                        usuarios_ignorados += 1
                        continue
                    
                    # Limitar exemplos por usuário
                    n_exemplos_positivos = min(len(historico_valido), max_exemplos_por_usuario // 2)
                    historico_amostrado = np.random.choice(
                        historico_valido, 
                        n_exemplos_positivos, 
                        replace=False
                    )
                    
                    # Adicionar exemplos positivos
                    for item_idx in historico_amostrado:
                        X_usuario_list.append(usuario_idx)
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(1)
                        total_exemplos += 1
                    
                    # Adicionar exemplos negativos balanceados
                    itens_negativos = np.random.choice(
                        [i for i in self.features_item.keys() if i not in historico],
                        size=len(historico_amostrado),
                        replace=False
                    )
                    
                    for item_idx in itens_negativos:
                        X_usuario_list.append(usuario_idx)
                        X_item_list.append(item_idx)
                        X_conteudo_list.append(self.features_item[item_idx]['vetor_conteudo'])
                        y_list.append(0)
                        total_exemplos += 1
                    
                    usuarios_processados += 1
                    if usuarios_processados % 100 == 0:
                        logger.info(f"Processados {usuarios_processados} usuários, "
                                  f"ignorados {usuarios_ignorados}, "
                                  f"total exemplos {total_exemplos}")
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar usuário {usuario_id}: {str(e)}")
                    usuarios_ignorados += 1
                    continue
            
            if not X_usuario_list:
                raise ValueError("Nenhum exemplo de treino gerado após processamento")
            
            # Converter para arrays numpy
            X_usuario = np.array(X_usuario_list, dtype=np.int32)
            X_item = np.array(X_item_list, dtype=np.int32)
            X_conteudo = np.array(X_conteudo_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            
            # Log final
            logger.info("\n=== Estatísticas Finais ===")
            logger.info(f"Total de exemplos gerados: {len(y)}")
            logger.info(f"Usuários processados: {usuarios_processados}")
            logger.info(f"Usuários ignorados: {usuarios_ignorados}")
            logger.info(f"Exemplos positivos: {np.sum(y == 1)}")
            logger.info(f"Exemplos negativos: {np.sum(y == 0)}")
            
            # Verificações de integridade
            max_usuario_idx = np.max(X_usuario)
            if max_usuario_idx >= len(self.usuario_id_to_index):
                raise ValueError(f"Índice de usuário {max_usuario_idx} maior que o número de usuários {len(self.usuario_id_to_index)}")
            
            # Salvar checkpoint
            self._salvar_checkpoint_exemplos((X_usuario, X_item, X_conteudo, y), caminho_checkpoints)
            
            return X_usuario, X_item, X_conteudo, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados de treino: {str(e)}")
            raise

    def prever(self, usuario_id, candidatos=None, k=10):
        """
        Faz previsões para um usuário.
        """
        if not self.modelo:
            raise ValueError("Modelo não treinado")
        
        try:
            usuario_id = str(usuario_id)
            if usuario_id not in self.usuario_id_to_index:
                logger.warning(f"Usuário {usuario_id} não encontrado no conjunto de treino")
                return []
            
            if not candidatos:
                candidatos = list(self.features_item.keys())
            
            usuario_idx = self.usuario_id_to_index[usuario_id]
            X_usuario = np.array([usuario_idx] * len(candidatos))
            X_item = np.array(candidatos)
            X_conteudo = np.array([self.features_item[idx]['vetor_conteudo'] for idx in candidatos])
            
            # Fazer previsões em lotes para economizar memória
            batch_size = 1000
            previsoes = []
            
            for i in range(0, len(candidatos), batch_size):
                batch_end = min(i + batch_size, len(candidatos))
                batch_previsoes = self.modelo.predict(
                    [
                        X_usuario[i:batch_end],
                        X_item[i:batch_end],
                        X_conteudo[i:batch_end]
                    ],
                    verbose=0
                )
                previsoes.extend(batch_previsoes.flatten())
            
            # Ordenar e retornar os top-k itens
            previsoes = np.array(previsoes)
            indices_ordenados = np.argsort(previsoes)[::-1][:k]
            
            recomendacoes = []
            for idx in indices_ordenados:
                item_id = self.index_to_item_id[candidatos[idx]]
                score = float(previsoes[idx])
                recomendacoes.append({
                    'item_id': item_id,
                    'score': score
                })
            
            return recomendacoes
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões: {str(e)}")
            return []

    def salvar_modelo(self, caminho):
        """
        Salva o modelo em disco.
        """
        logger.info(f"Salvando modelo em {caminho}")
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(caminho), exist_ok=True)
            
            dados_modelo = {
                'modelo': self.modelo,
                'tfidf': self.tfidf,
                'item_id_to_index': self.item_id_to_index,
                'index_to_item_id': self.index_to_item_id,
                'usuario_id_to_index': self.usuario_id_to_index,
                'index_to_usuario_id': self.index_to_usuario_id,
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
            instancia.usuario_id_to_index = dados_modelo['usuario_id_to_index']
            instancia.index_to_usuario_id = dados_modelo['index_to_usuario_id']
            instancia.item_count = dados_modelo['item_count']
            instancia.itens_usuario = dados_modelo['itens_usuario']
            instancia.features_item = dados_modelo['features_item']
            
            logger.info("Modelo carregado com sucesso")
            return instancia
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise