from pyspark.sql import SparkSession, DataFrame
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
            self.spark.stop()