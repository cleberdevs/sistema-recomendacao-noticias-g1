'''from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, explode, from_json, to_timestamp, 
                                 concat, lit, udf, count, min, max, coalesce, size, avg, length, when)
from pyspark.sql.types import (ArrayType, StringType, TimestampType, 
                              StructType, StructField)
import logging
from typing import Tuple
import json
import pandas as pd
from src.config.spark_config import criar_spark_session, configurar_log_nivel

logger = logging.getLogger(__name__)

def _processar_lista(valor: str) -> list:
    """Processa string JSON para lista."""
    try:
        if valor:
            return json.loads(valor)
        return []
    except Exception as e:
        logger.error(f"Erro ao processar lista: {str(e)}")
        return []

class PreProcessadorDadosSpark:
    def __init__(self, memoria_executor="4g", memoria_driver="4g"):
        """
        Inicializa o preprocessador com Spark.
        
        Args:
            memoria_executor: Quantidade de memória para executores Spark
            memoria_driver: Quantidade de memória para o driver Spark
        """
        self.spark = criar_spark_session(
            memoria_executor=memoria_executor,
            memoria_driver=memoria_driver
        )
        configurar_log_nivel(self.spark)
        
        logger.info("Sessão Spark inicializada")
        
        # Definir schemas para melhor performance
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])

    def processar_dados_treino(self, arquivos_treino: list, 
                             arquivos_itens: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa os dados usando Spark.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames pandas de treino e itens processados
        """
        logger.info("Iniciando processamento dos dados com Spark")
        
        try:
            # Registrar UDF para processamento de listas
            processar_lista_udf = udf(_processar_lista, ArrayType(StringType()))
            
            # Carregar e processar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            ).cache()  # Cache para reuso
            
            logger.info(f"Registros de treino carregados: {df_treino.count()}")
            
            # Processar colunas de listas
            df_treino = df_treino.withColumn(
                "historico", 
                processar_lista_udf(col("history"))
            ).withColumn(
                "historicoTimestamp", 
                processar_lista_udf(col("timestampHistory"))
            )
            
            # Renomear e selecionar colunas
            df_treino = df_treino.select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            )
            
            # Carregar e processar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).cache()
            
            logger.info(f"Registros de itens carregados: {df_itens.count()}")
            
            # Processar itens
            df_itens = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao")
            
            # Processar features de texto
            logger.info("Processando features de texto")
            df_itens = df_itens.withColumn(
                "conteudo_texto",
                concat(
                    coalesce(col("Titulo"), lit("")),
                    lit(" "),
                    coalesce(col("Corpo"), lit("")),
                    lit(" "),
                    coalesce(col("Caption"), lit(""))
                )
            )
            
            # Converter para pandas
            logger.info("Convertendo para pandas")
            dados_treino = df_treino.toPandas()
            dados_itens = df_itens.toPandas()
            
            # Liberar cache
            df_treino.unpersist()
            df_itens.unpersist()
            
            logger.info("Processamento com Spark concluído")
            return dados_treino, dados_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento Spark: {str(e)}")
            raise

    def validar_dados(self, dados_treino: pd.DataFrame, 
                     dados_itens: pd.DataFrame) -> bool:
        """
        Validação dos dados processados usando Spark para eficiência.
        
        Args:
            dados_treino: DataFrame pandas com dados de treino
            dados_itens: DataFrame pandas com dados de itens
            
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Validando dados processados")
        try:
            # Converter para Spark DataFrames para validação eficiente
            df_treino = self.spark.createDataFrame(dados_treino)
            df_itens = self.spark.createDataFrame(dados_itens)
            
            # Cache para múltiplas operações
            df_treino.cache()
            df_itens.cache()
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).toPandas()
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).toPandas()
            
            tem_nulos = False
            if nulos_treino.values.sum() > 0:
                logger.warning("Valores nulos encontrados nos dados de treino:")
                for coluna in df_treino.columns:
                    if nulos_treino[coluna].values[0] > 0:
                        logger.warning(f"{coluna}: {nulos_treino[coluna].values[0]} nulos")
                tem_nulos = True
                
            if nulos_itens.values.sum() > 0:
                logger.warning("Valores nulos encontrados nos dados de itens:")
                for coluna in df_itens.columns:
                    if nulos_itens[coluna].values[0] > 0:
                        logger.warning(f"{coluna}: {nulos_itens[coluna].values[0]} nulos")
                tem_nulos = True
            
            # Verificar consistência entre histórico e itens
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct()
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            )
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico que não existem nos dados de itens"
                )
                
                # Mostrar alguns exemplos
                logger.warning("Exemplos de itens faltantes:")
                itens_faltantes.show(5, truncate=False)
            
            # Verificar timestamps
            logger.info("Verificando timestamps")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Liberar cache
            df_treino.unpersist()
            df_itens.unpersist()
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False

    def mostrar_info_dados(self, dados_treino: pd.DataFrame, 
                          dados_itens: pd.DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            dados_treino: DataFrame pandas com dados de treino
            dados_itens: DataFrame pandas com dados de itens
        """
        try:
            # Converter para Spark DataFrames
            df_treino = self.spark.createDataFrame(dados_treino)
            df_itens = self.spark.createDataFrame(dados_itens)
            
            # Cache para múltiplas operações
            df_treino.cache()
            df_itens.cache()
            
            print("\nInformações dos dados de treino:")
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            print(f"Número de registros: {n_registros}")
            print(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            print("\nEstatísticas do histórico:")
            print(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            print(f"Mínimo de itens: {tamanho_historico['minimo']}")
            print(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            print("\nInformações dos dados de itens:")
            n_itens = df_itens.count()
            print(f"Número de itens: {n_itens}")
            
            print("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            print("\nEstatísticas do conteúdo:")
            print(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            print(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            print(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
            # Liberar cache
            df_treino.unpersist()
            df_itens.unpersist()
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise

    def __del__(self):
        """Encerra a sessão Spark ao finalizar."""
        if hasattr(self, 'spark'):
            self.spark.stop()'''

'''from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, explode, from_json, to_timestamp, 
                                 concat, lit, udf, count, min, max, coalesce, size, avg, length, when)
from pyspark.sql.types import (ArrayType, StringType, TimestampType, 
                              StructType, StructField)
import logging
from typing import Tuple
import json
import pandas as pd
from src.config.spark_config import criar_spark_session, configurar_log_nivel

logger = logging.getLogger(__name__)

def _processar_lista(valor: str) -> list:
    """Processa string JSON para lista."""
    try:
        if valor:
            return json.loads(valor)
        return []
    except Exception as e:
        logger.error(f"Erro ao processar lista: {str(e)}")
        return []

class PreProcessadorDadosSpark:
    def __init__(self, memoria_executor="4g", memoria_driver="4g"):
        """
        Inicializa o preprocessador com Spark.
        
        Args:
            memoria_executor: Quantidade de memória para executores Spark
            memoria_driver: Quantidade de memória para o driver Spark
        """
        self.spark = criar_spark_session(
            memoria_executor=memoria_executor,
            memoria_driver=memoria_driver
        )
        configurar_log_nivel(self.spark)
        
        logger.info("Sessão Spark inicializada")
        
        # Definir schemas para melhor performance
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])

    def processar_dados_treino(self, arquivos_treino: list, 
                             arquivos_itens: list) -> Tuple[DataFrame, DataFrame]:
        """
        Processa os dados usando Spark.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames Spark de treino e itens processados
        """
        logger.info("Iniciando processamento dos dados com Spark")
        
        try:
            # Carregar e processar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            ).cache()  # Cache para reuso
            
            logger.info(f"Registros de treino carregados: {df_treino.count()}")
            
            # Processar colunas de listas usando from_json (nativo do Spark)
            df_treino = df_treino.withColumn(
                "historico", 
                from_json(col("history"), ArrayType(StringType()))
            ).withColumn(
                "historicoTimestamp", 
                from_json(col("timestampHistory"), ArrayType(StringType()))
            )
            
            # Renomear e selecionar colunas
            df_treino = df_treino.select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            )
            
            # Carregar e processar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).cache()
            
            logger.info(f"Registros de itens carregados: {df_itens.count()}")
            
            # Processar itens
            df_itens = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao")
            
            # Processar features de texto
            logger.info("Processando features de texto")
            df_itens = df_itens.withColumn(
                "conteudo_texto",
                concat(
                    coalesce(col("Titulo"), lit("")),
                    lit(" "),
                    coalesce(col("Corpo"), lit("")),
                    lit(" "),
                    coalesce(col("Caption"), lit(""))
                )
            )
            
            logger.info("Processamento com Spark concluído")
            return df_treino, df_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento Spark: {str(e)}")
            raise
        finally:
            # Liberar cache em caso de erro
            if 'df_treino' in locals():
                df_treino.unpersist()
            if 'df_itens' in locals():
                df_itens.unpersist()

    def validar_dados(self, df_treino: DataFrame, 
                     df_itens: DataFrame) -> bool:
        """
        Validação dos dados processados usando Spark para eficiência.
        
        Args:
            df_treino: DataFrame Spark com dados de treino
            df_itens: DataFrame Spark com dados de itens
            
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Validando dados processados")
        try:
            # Cache para múltiplas operações
            df_treino.cache()
            df_itens.cache()
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).collect()[0]
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).collect()[0]
            
            tem_nulos = False
            for coluna, nulos in nulos_treino.asDict().items():
                if nulos > 0:
                    logger.warning(f"{coluna}: {nulos} nulos")
                    tem_nulos = True
                    
            for coluna, nulos in nulos_itens.asDict().items():
                if nulos > 0:
                    logger.warning(f"{coluna}: {nulos} nulos")
                    tem_nulos = True
            
            # Verificar consistência entre histórico e itens
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct()
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            )
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico que não existem nos dados de itens"
                )
                
                # Mostrar alguns exemplos
                logger.warning("Exemplos de itens faltantes:")
                itens_faltantes.show(5, truncate=False)
            
            # Verificar timestamps
            logger.info("Verificando timestamps")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
        finally:
            # Liberar cache
            if 'df_treino' in locals():
                df_treino.unpersist()
            if 'df_itens' in locals():
                df_itens.unpersist()

    def mostrar_info_dados(self, df_treino: DataFrame, 
                          df_itens: DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            df_treino: DataFrame Spark com dados de treino
            df_itens: DataFrame Spark com dados de itens
        """
        try:
            # Cache para múltiplas operações
            df_treino.cache()
            df_itens.cache()
            
            print("\nInformações dos dados de treino:")
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            print(f"Número de registros: {n_registros}")
            print(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            print("\nEstatísticas do histórico:")
            print(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            print(f"Mínimo de itens: {tamanho_historico['minimo']}")
            print(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            print("\nInformações dos dados de itens:")
            n_itens = df_itens.count()
            print(f"Número de itens: {n_itens}")
            
            print("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            print("\nEstatísticas do conteúdo:")
            print(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            print(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            print(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise
        finally:
            # Liberar cache
            if 'df_treino' in locals():
                df_treino.unpersist()
            if 'df_itens' in locals():
                df_itens.unpersist()

    def __del__(self):
        """Encerra a sessão Spark ao finalizar."""
        if hasattr(self, 'spark'):
            self.spark.stop()'''

'''from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, explode, from_json, to_timestamp, concat, lit, udf,
    count, min, max, coalesce, size, avg, length, when
)
from pyspark.sql.types import ArrayType, StringType, TimestampType, StructType, StructField
import logging
from typing import Tuple, Optional, List
import json

logger = logging.getLogger(__name__)

class PreProcessadorDadosSpark:
    def __init__(self, spark: SparkSession):
        """
        Inicializa o preprocessador com uma sessão Spark.
        
        Args:
            spark: Sessão Spark ativa
        """
        self.spark = spark
        
        # Definir schemas otimizados
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])
        
        logger.info("PreProcessador inicializado com sucesso")
    
    def _verificar_spark_ativo(self):
        """Verifica se a sessão Spark está ativa."""
        if not self.spark or self.spark._jsc.sc().isStopped():
            raise RuntimeError("Sessão Spark não está ativa")
    
    def _processar_lista_json(self, valor: str) -> list:
        """
        Processa string JSON para lista.
        
        Args:
            valor: String JSON a ser processada
            
        Returns:
            Lista processada
        """
        try:
            if valor and isinstance(valor, str):
                return json.loads(valor)
            return []
        except Exception as e:
            logger.error(f"Erro ao processar JSON: {str(e)}")
            return []

    def processar_dados_treino(
        self, 
        arquivos_treino: List[str], 
        arquivos_itens: List[str]
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Processa os dados de treino e itens.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames processados de treino e itens
        """
        logger.info("Iniciando processamento dos dados")
        self._verificar_spark_ativo()
        
        df_treino = None
        df_itens = None
        
        try:
            # Processar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            ).checkpoint()
            
            num_registros_treino = df_treino.count()
            logger.info(f"Registros de treino carregados: {num_registros_treino}")
            
            if num_registros_treino == 0:
                raise ValueError("Nenhum dado de treino encontrado")
            
            # Processar colunas JSON
            df_treino = df_treino.withColumn(
                "historico", 
                from_json(col("history"), ArrayType(StringType()))
            ).withColumn(
                "historicoTimestamp", 
                from_json(col("timestampHistory"), ArrayType(StringType()))
            ).select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            ).persist()
            
            # Processar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).checkpoint()
            
            num_registros_itens = df_itens.count()
            logger.info(f"Registros de itens carregados: {num_registros_itens}")
            
            if num_registros_itens == 0:
                raise ValueError("Nenhum dado de item encontrado")
            
            # Processar colunas dos itens
            df_itens = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao")
            
            # Criar coluna de texto combinado
            df_itens = df_itens.withColumn(
                "conteudo_texto",
                concat(
                    coalesce(col("Titulo"), lit("")),
                    lit(" "),
                    coalesce(col("Corpo"), lit("")),
                    lit(" "),
                    coalesce(col("Caption"), lit(""))
                )
            ).persist()
            
            return df_treino, df_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            # Limpar recursos em caso de erro
            for df in [df_treino, df_itens]:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as cleanup_error:
                        logger.error(f"Erro ao limpar DataFrame: {str(cleanup_error)}")
            raise

    def validar_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> bool:
        """
        Valida a qualidade dos dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
            
        Returns:
            bool indicando se os dados são válidos
        """
        logger.info("Validando dados processados")
        self._verificar_spark_ativo()
        
        try:
            # Cache temporário para validação
            df_treino.cache()
            df_itens.cache()
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).collect()[0]
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).collect()[0]
            
            tem_nulos = False
            for coluna, nulos in nulos_treino.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
                    
            for coluna, nulos in nulos_itens.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
            
            # Verificar consistência
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct()
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            )
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico não encontrados nos dados de itens"
                )
                itens_faltantes.show(5, truncate=False)
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
        finally:
            # Limpar cache
            for df in [df_treino, df_itens]:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def mostrar_info_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
        """
        try:
            self._verificar_spark_ativo()
            
            # Cache temporário
            df_treino.cache()
            df_itens.cache()
            
            # Informações gerais
            logger.info("\nInformações dos dados de treino:")
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            logger.info(f"Número de registros: {n_registros}")
            logger.info(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do histórico:")
            logger.info(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            logger.info(f"Mínimo de itens: {tamanho_historico['minimo']}")
            logger.info(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            # Informações dos itens
            logger.info("\nInformações dos dados de itens:")
            n_itens = df_itens.count()
            logger.info(f"Número de itens: {n_itens}")
            
            # Período dos dados
            logger.info("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do conteúdo:")
            logger.info(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            logger.info(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            logger.info(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
        finally:
            # Limpar cache
            for df in [df_treino, df_itens]:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")'''

'''from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, explode, from_json, to_timestamp, concat_ws, lit, 
    count, min, max, coalesce, size, avg, length, when
)
from pyspark.sql.types import ArrayType, StringType, TimestampType, StructType, StructField
import logging
from typing import Tuple, Optional
import json

logger = logging.getLogger(__name__)

class PreProcessadorDadosSpark:
    def __init__(self, spark: SparkSession):
        """
        Inicializa o preprocessador com uma sessão Spark.
        
        Args:
            spark: Sessão Spark ativa
        """
        self.spark = spark
        
        # Definir schemas otimizados
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])
        
        logger.info("PreProcessador inicializado")

    def _verificar_spark_ativo(self):
        """Verifica se a sessão Spark está ativa."""
        if not self.spark or self.spark._jsc.sc().isStopped():
            raise RuntimeError("Sessão Spark não está ativa")

    def processar_dados_treino(
        self, 
        arquivos_treino: list, 
        arquivos_itens: list
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Processa os dados usando Spark.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames processados de treino e itens
        """
        logger.info("Iniciando processamento dos dados")
        self._verificar_spark_ativo()
        
        try:
            # Carregar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            )
            
            # Processar colunas JSON
            df_treino = df_treino.withColumn(
                "historico", 
                from_json(col("history"), ArrayType(StringType()))
            ).withColumn(
                "historicoTimestamp", 
                from_json(col("timestampHistory"), ArrayType(StringType()))
            ).select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            ).persist()
            
            num_registros_treino = df_treino.count()
            logger.info(f"Registros de treino processados: {num_registros_treino}")
            
            # Carregar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            )
            
            # Processar itens
            df_itens = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao")
            
            # Criar coluna de texto combinado
            df_itens = df_itens.withColumn(
                "conteudo_texto",
                concat_ws(
                    " ",
                    coalesce(col("Titulo"), lit("")),
                    coalesce(col("Corpo"), lit("")),
                    coalesce(col("Caption"), lit(""))
                )
            ).persist()
            
            num_registros_itens = df_itens.count()
            logger.info(f"Registros de itens processados: {num_registros_itens}")
            
            return df_treino, df_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            raise

    def validar_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> bool:
        """
        Valida a qualidade dos dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
            
        Returns:
            bool indicando se os dados são válidos
        """
        logger.info("Validando dados processados")
        self._verificar_spark_ativo()
        
        try:
            # Cache para múltiplas operações
            df_treino.cache()
            df_itens.cache()
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).collect()[0]
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).collect()[0]
            
            tem_nulos = False
            for coluna, nulos in nulos_treino.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
                    
            for coluna, nulos in nulos_itens.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
            
            # Verificar consistência
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct()
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            )
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico não encontrados nos dados de itens"
                )
                itens_faltantes.show(5, truncate=False)
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
        finally:
            # Liberar cache
            if df_treino.is_cached:
                df_treino.unpersist()
            if df_itens.is_cached:
                df_itens.unpersist()

    def mostrar_info_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
        """
        try:
            self._verificar_spark_ativo()
            
            # Cache temporário
            df_treino.cache()
            df_itens.cache()
            
            # Informações gerais
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            logger.info("\nInformações dos dados de treino:")
            logger.info(f"Número de registros: {n_registros}")
            logger.info(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do histórico:")
            logger.info(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            logger.info(f"Mínimo de itens: {tamanho_historico['minimo']}")
            logger.info(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            # Informações dos itens
            n_itens = df_itens.count()
            logger.info(f"\nNúmero de itens: {n_itens}")
            
            # Período dos dados
            logger.info("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do conteúdo:")
            logger.info(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            logger.info(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            logger.info(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise
        finally:
            # Liberar cache
            if df_treino.is_cached:
                df_treino.unpersist()
            if df_itens.is_cached:
                df_itens.unpersist()'''

'''from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, explode, from_json, to_timestamp, concat_ws, lit, 
    count, min, max, coalesce, size, avg, length, when
)
from pyspark.sql.types import ArrayType, StringType, TimestampType, StructType, StructField
import logging
from typing import Tuple, Optional
import time

logger = logging.getLogger(__name__)

class PreProcessadorDadosSpark:
    def __init__(self, spark: SparkSession):
        """
        Inicializa o preprocessador com uma sessão Spark.
        
        Args:
            spark: Sessão Spark ativa
        """
        self.spark = spark
        
        # Definir schemas otimizados
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])
        
        logger.info("PreProcessador inicializado")
        
    def _verificar_spark_ativo(self):
        """Verifica se a sessão Spark está ativa."""
        if not self.spark or self.spark._jsc.sc().isStopped():
            raise RuntimeError("Sessão Spark não está ativa")

    def processar_dados_treino(
        self, 
        arquivos_treino: list, 
        arquivos_itens: list
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Processa os dados usando Spark.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames processados de treino e itens
        """
        logger.info("Iniciando processamento dos dados")
        self._verificar_spark_ativo()
        
        try:
            # Carregar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            ).cache()
            
            # Forçar avaliação
            num_registros_treino = df_treino.count()
            logger.info(f"Registros de treino carregados: {num_registros_treino}")
            
            # Processar colunas JSON
            df_treino = df_treino.withColumn(
                "historico", 
                from_json(col("history"), ArrayType(StringType()))
            ).withColumn(
                "historicoTimestamp", 
                from_json(col("timestampHistory"), ArrayType(StringType()))
            ).select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            ).checkpoint()  # Persistir transformação
            
            # Carregar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).cache()
            
            # Forçar avaliação
            num_registros_itens = df_itens.count()
            logger.info(f"Registros de itens carregados: {num_registros_itens}")
            
            # Processar itens
            df_itens = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao")
            
            # Criar coluna de texto combinado
            df_itens = df_itens.withColumn(
                "conteudo_texto",
                concat_ws(
                    " ",
                    coalesce(col("Titulo"), lit("")),
                    coalesce(col("Corpo"), lit("")),
                    coalesce(col("Caption"), lit(""))
                )
            ).checkpoint()  # Persistir transformação
            
            return df_treino, df_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            raise

    def validar_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> bool:
        """
        Valida a qualidade dos dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
            
        Returns:
            bool indicando se os dados são válidos
        """
        logger.info("Validando dados processados")
        self._verificar_spark_ativo()
        
        try:
            # Cache para operações de validação
            df_treino.cache()
            df_itens.cache()
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).collect()[0]
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).collect()[0]
            
            tem_nulos = False
            for coluna, nulos in nulos_treino.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
                    
            for coluna, nulos in nulos_itens.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
            
            # Verificar consistência
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct()
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            ).cache()
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico não encontrados nos dados de itens"
                )
                itens_faltantes.show(5, truncate=False)
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
        finally:
            # Limpar cache
            for df in [df_treino, df_itens, itens_faltantes]:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def mostrar_info_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
        """
        try:
            self._verificar_spark_ativo()
            
            # Cache temporário
            df_treino.cache()
            df_itens.cache()
            
            # Informações gerais
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            logger.info("\nInformações dos dados de treino:")
            logger.info(f"Número de registros: {n_registros}")
            logger.info(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do histórico:")
            logger.info(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            logger.info(f"Mínimo de itens: {tamanho_historico['minimo']}")
            logger.info(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            # Informações dos itens
            n_itens = df_itens.count()
            logger.info(f"\nNúmero de itens: {n_itens}")
            
            # Período dos dados
            logger.info("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do conteúdo:")
            logger.info(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            logger.info(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            logger.info(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise
        finally:
            # Limpar cache
            for df in [df_treino, df_itens]:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")'''

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, explode, from_json, to_timestamp, concat_ws, lit, 
    count, min, max, coalesce, size, avg, length, when
)
from pyspark.sql.types import (
    ArrayType, StringType, TimestampType, StructType, StructField
)
import logging
from typing import Tuple, Optional
import os
import time
import shutil

logger = logging.getLogger(__name__)

class PreProcessadorDadosSpark:
    """
    Classe para preprocessamento de dados usando PySpark.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Inicializa o preprocessador com uma sessão Spark.
        
        Args:
            spark: Sessão Spark ativa
        """
        self.spark = spark
        
        # Schema para dados de treino
        self.schema_treino = StructType([
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True)
        ])
        
        # Schema para dados de itens
        self.schema_itens = StructType([
            StructField("Page", StringType(), True),
            StructField("Title", StringType(), True),
            StructField("Body", StringType(), True),
            StructField("Issued", TimestampType(), True),
            StructField("Modified", TimestampType(), True),
            StructField("Caption", StringType(), True)
        ])
        
        # Configurar diretório de checkpoint
        self._configurar_checkpoints()
        
        logger.info("PreProcessador inicializado")

    def _configurar_checkpoints(self):
        """Configura diretório de checkpoints."""
        try:
            checkpoint_dir = "checkpoints"
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
            logger.info("Diretório de checkpoints configurado")
        except Exception as e:
            logger.error(f"Erro ao configurar checkpoints: {str(e)}")
            raise

    def _verificar_spark_ativo(self):
        """Verifica se a sessão Spark está ativa."""
        if not self.spark or self.spark._jsc.sc().isStopped():
            raise RuntimeError("Sessão Spark não está ativa")

    def processar_dados_treino(
        self, 
        arquivos_treino: list, 
        arquivos_itens: list
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Processa os dados usando Spark.
        
        Args:
            arquivos_treino: Lista de caminhos dos arquivos de treino
            arquivos_itens: Lista de caminhos dos arquivos de itens
            
        Returns:
            Tuple contendo DataFrames processados de treino e itens
        """
        logger.info("Iniciando processamento dos dados")
        self._verificar_spark_ativo()
        
        dfs_to_unpersist = []
        
        try:
            # Processar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=self.schema_treino
            ).persist()
            dfs_to_unpersist.append(df_treino)
            
            # Forçar avaliação
            num_registros_treino = df_treino.count()
            logger.info(f"Registros de treino carregados: {num_registros_treino}")
            
            # Processar colunas JSON
            df_treino_processado = df_treino.withColumn(
                "historico", 
                from_json(col("history"), ArrayType(StringType()))
            ).withColumn(
                "historicoTimestamp", 
                from_json(col("timestampHistory"), ArrayType(StringType()))
            ).select(
                col("historico"),
                col("historicoTimestamp"),
                col("userId").alias("idUsuario")
            ).persist()
            dfs_to_unpersist.append(df_treino_processado)
            
            # Processar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).persist()
            dfs_to_unpersist.append(df_itens)
            
            # Forçar avaliação
            num_registros_itens = df_itens.count()
            logger.info(f"Registros de itens carregados: {num_registros_itens}")
            
            # Processar itens
            df_itens_processado = df_itens.withColumnRenamed("Page", "Pagina") \
                              .withColumnRenamed("Title", "Titulo") \
                              .withColumnRenamed("Body", "Corpo") \
                              .withColumnRenamed("Issued", "DataPublicacao") \
                              .withColumn(
                                  "conteudo_texto",
                                  concat_ws(
                                      " ",
                                      coalesce(col("Titulo"), lit("")),
                                      coalesce(col("Corpo"), lit("")),
                                      coalesce(col("Caption"), lit(""))
                                  )
                              ).persist()
            dfs_to_unpersist.append(df_itens_processado)
            
            # Liberar DataFrames intermediários
            df_treino.unpersist()
            df_itens.unpersist()
            
            return df_treino_processado, df_itens_processado
            
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            raise
        finally:
            # Limpar cache de DataFrames intermediários
            for df in dfs_to_unpersist:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def validar_dados(
        self, 
        df_treino: DataFrame, 
        df_itens: DataFrame
    ) -> bool:
        """
        Valida a qualidade dos dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
            
        Returns:
            bool indicando se os dados são válidos
        """
        logger.info("Validando dados processados")
        self._verificar_spark_ativo()
        
        dfs_to_unpersist = []
        
        try:
            # Cache para operações de validação
            df_treino.persist()
            df_itens.persist()
            dfs_to_unpersist.extend([df_treino, df_itens])
            
            # Verificar valores nulos
            logger.info("Verificando valores nulos")
            nulos_treino = df_treino.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_treino.columns
            ]).collect()[0]
            
            nulos_itens = df_itens.select([
                count(when(col(c).isNull(), c)).alias(c) 
                for c in df_itens.columns
            ]).collect()[0]
            
            tem_nulos = False
            
            # Verificar nulos nos dados de treino
            for coluna, nulos in nulos_treino.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
            
            # Verificar nulos nos dados de itens
            for coluna, nulos in nulos_itens.asDict().items():
                if nulos > 0:
                    logger.warning(f"Coluna {coluna}: {nulos} valores nulos")
                    tem_nulos = True
            
            # Verificar consistência entre histórico e itens
            logger.info("Verificando consistência dos dados")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct().persist()
            dfs_to_unpersist.append(historico_items)
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.Pagina,
                "left_anti"
            ).persist()
            dfs_to_unpersist.append(itens_faltantes)
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico não encontrados nos dados de itens"
                )
                itens_faltantes.show(5, truncate=False)
            
            return not tem_nulos
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
        finally:
            # Limpar cache
            for df in dfs_to_unpersist:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def mostrar_info_dados(
        self, 
        df_treino: DataFrame, 
        df_itens: DataFrame
    ) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        
        Args:
            df_treino: DataFrame com dados de treino
            df_itens: DataFrame com dados dos itens
        """
        try:
            self._verificar_spark_ativo()
            
            dfs_to_unpersist = []
            
            # Cache temporário para análise
            df_treino.persist()
            df_itens.persist()
            dfs_to_unpersist.extend([df_treino, df_itens])
            
            # Informações gerais
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            logger.info("\nInformações dos dados de treino:")
            logger.info(f"Número de registros: {n_registros}")
            logger.info(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            tamanho_historico = df_treino.select(
                size("historico").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do histórico:")
            logger.info(f"Média de itens por usuário: {tamanho_historico['media']:.2f}")
            logger.info(f"Mínimo de itens: {tamanho_historico['minimo']}")
            logger.info(f"Máximo de itens: {tamanho_historico['maximo']}")
            
            # Informações dos itens
            n_itens = df_itens.count()
            logger.info(f"\nNúmero de itens: {n_itens}")
            
            # Período dos dados
            logger.info("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Estatísticas do conteúdo
            tamanho_conteudo = df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).agg(
                avg("tamanho").alias("media"),
                min("tamanho").alias("minimo"),
                max("tamanho").alias("maximo")
            ).collect()[0]
            
            logger.info("\nEstatísticas do conteúdo:")
            logger.info(f"Tamanho médio do texto: {tamanho_conteudo['media']:.2f} caracteres")
            logger.info(f"Menor texto: {tamanho_conteudo['minimo']} caracteres")
            logger.info(f"Maior texto: {tamanho_conteudo['maximo']} caracteres")
            
            # Distribuição de tamanhos de texto
            logger.info("\nDistribuição de tamanhos de texto:")
            df_itens.select(
                length("conteudo_texto").alias("tamanho")
            ).describe().show()
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise
        finally:
            # Limpar cache
            for df in dfs_to_unpersist:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def limpar_recursos(self):
        """Limpa recursos e cache do Spark."""
        try:
            # Limpar cache do Spark
            self.spark.catalog.clearCache()
            
            # Limpar diretório de checkpoint
            self._configurar_checkpoints()
            
            logger.info("Recursos limpos com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao limpar recursos: {str(e)}")