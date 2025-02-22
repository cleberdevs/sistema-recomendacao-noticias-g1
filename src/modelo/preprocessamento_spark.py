from ast import expr
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, explode, from_json, to_timestamp, concat_ws, lit, 
    count, min, max, coalesce, size, avg, length, when, array,
    year, split, regexp_replace, element_at, array_remove
)
from pyspark.sql.types import (
    ArrayType, StringType, TimestampType, StructType, StructField,
    IntegerType, DoubleType
)
import logging
from typing import Tuple, Optional
import os
import time
import shutil
from datetime import datetime
import json



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
        
        # Schema atualizado para dados de treino
        self.schema_treino = StructType([
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("historySize", IntegerType(), True),
            StructField("history", ArrayType(StringType()), True),
            StructField("timestampHistory", ArrayType(TimestampType()), True),
            StructField("numberOfClicksHistory", ArrayType(IntegerType()), True),
            StructField("timeOnPageHistory", ArrayType(DoubleType()), True),
            StructField("scrollPercentageHistory", ArrayType(DoubleType()), True),
            StructField("pageVisitsCountHistory", ArrayType(IntegerType()), True),
            StructField("timestampHistory_new", ArrayType(TimestampType()), True)
        ])
        
        # Schema atualizado para dados de itens
        self.schema_itens = StructType([
            StructField("page", StringType(), True),
            StructField("url", StringType(), True),
            StructField("issued", TimestampType(), True),
            StructField("modified", TimestampType(), True),
            StructField("title", StringType(), True),
            StructField("body", StringType(), True),
            StructField("caption", StringType(), True)
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
        Processa os dados usando Spark e salva os resultados.
        """
        logger.info("Iniciando processamento dos dados")
        self._verificar_spark_ativo()
        
        # Configurar modo de escrita para datas antigas
        self.spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
        
        # Schema modificado para ler strings primeiro
        schema_treino = StructType([
            StructField("userId", StringType(), True),
            StructField("userType", StringType(), True),
            StructField("historySize", StringType(), True),
            StructField("history", StringType(), True),
            StructField("timestampHistory", StringType(), True),
            StructField("numberOfClicksHistory", StringType(), True),
            StructField("timeOnPageHistory", StringType(), True),
            StructField("scrollPercentageHistory", StringType(), True),
            StructField("pageVisitsCountHistory", StringType(), True),
            StructField("timestampHistory_new", StringType(), True)
        ])
        
        dfs_to_unpersist = []
        
        try:
            # Carregar dados de treino
            logger.info("Carregando dados de treino")
            df_treino = self.spark.read.csv(
                arquivos_treino, 
                header=True, 
                schema=schema_treino
            ).persist()
            dfs_to_unpersist.append(df_treino)
            
            # Carregar dados dos itens
            logger.info("Carregando dados dos itens")
            df_itens = self.spark.read.csv(
                arquivos_itens, 
                header=True, 
                schema=self.schema_itens
            ).persist()
            
            # Definir limites de datas
            max_year = 2030
            min_year = 1970
            
            # Processar e limpar dados dos itens
            df_itens_processado = df_itens \
                .withColumn(
                    "DataPublicacao",
                    when(
                        (year(col("issued")) >= min_year) & 
                        (year(col("issued")) <= max_year),
                        col("issued")
                    ).otherwise(None)
                ) \
                .withColumn(
                    "conteudo_texto",
                    concat_ws(
                        " ",
                        coalesce(col("title"), lit("")),
                        coalesce(col("caption"), lit(""))
                    )
                ).persist()
            
            # Debug: mostrar dados brutos
            logger.info("Amostra dos dados brutos de treino:")
            df_treino.show(5, truncate=False)
            
            # Contar registros antes do processamento
            n_registros_inicial = df_treino.count()
            logger.info(f"Número inicial de registros: {n_registros_inicial}")
            
            # Verificar valores nulos no histórico
            n_historico_nulo = df_treino.filter(col("history").isNull()).count()
            logger.info(f"Registros com histórico nulo: {n_historico_nulo}")
            
            # Verificar formato do histórico
            logger.info("Exemplo de valores da coluna history:")
            df_treino.select("history").show(5, truncate=False)
            
            # Processar histórico
            df_treino_processado = df_treino
            
            # Método 1: Tentar como array JSON
            try:
                df_treino_processado = df_treino_processado.withColumn(
                    "history_array",
                    from_json(col("history"), ArrayType(StringType()))
                )
                logger.info("Processamento como JSON realizado")
            except Exception as e:
                logger.warning(f"Erro ao processar como JSON: {str(e)}")
            
            # Método 2: Tentar split por vírgula se for string
            try:
                df_treino_processado = df_treino_processado.withColumn(
                    "history_split",
                    split(
                        regexp_replace(
                            regexp_replace(col("history"), r"\[|\]", ""),
                            r"\s+", ""
                        ),
                        ","
                    )
                )
                logger.info("Processamento com split realizado")
            except Exception as e:
                logger.warning(f"Erro ao processar com split: {str(e)}")

            # Verificar resultados do processamento
            logger.info("Resultados do processamento do histórico:")
            df_treino_processado.select(
                "history",
                "history_array",
                "history_split"
            ).show(5, truncate=False)
            
            # Selecionar a melhor coluna processada
            if df_treino_processado.filter(size("history_array") > 0).count() > 0:
                historico_final = "history_array"
            elif df_treino_processado.filter(size("history_split") > 0).count() > 0:
                historico_final = "history_split"
            else:
                raise ValueError("Não foi possível processar o histórico em nenhum formato")

            # Processar o restante dos campos
            df_treino_final = df_treino_processado \
                .withColumn("historySize", col("historySize").cast(IntegerType())) \
                .withColumn("timestampHistory", from_json(col("timestampHistory"), ArrayType(TimestampType()))) \
                .withColumn("numberOfClicksHistory", from_json(col("numberOfClicksHistory"), ArrayType(IntegerType()))) \
                .withColumn("timeOnPageHistory", from_json(col("timeOnPageHistory"), ArrayType(DoubleType()))) \
                .withColumn("scrollPercentageHistory", from_json(col("scrollPercentageHistory"), ArrayType(DoubleType()))) \
                .withColumn("pageVisitsCountHistory", from_json(col("pageVisitsCountHistory"), ArrayType(IntegerType()))) \
                .select(
                    col("userId").alias("idUsuario"),
                    col("userType"),
                    col(historico_final).alias("historico"),
                    col("timestampHistory"),
                    col("numberOfClicksHistory"),
                    col("timeOnPageHistory"),
                    col("scrollPercentageHistory"),
                    col("pageVisitsCountHistory")
                ).persist()

            # Processar timestamps no histórico
            logger.info("Processando timestamps...")

            # Função para validar timestamps
            def validar_timestamp(timestamp_col):
                return when(
                    (year(timestamp_col) >= min_year) & 
                    (year(timestamp_col) <= max_year),
                    timestamp_col
                ).otherwise(lit(None))

            # Criar uma lista de expressões para cada possível posição no array
            max_timestamps = 50  # Número máximo de timestamps a considerar
            timestamp_expressions = []

            for i in range(max_timestamps):
                idx = lit(i + 1)
                timestamp_expressions.append(
                    validar_timestamp(element_at(col("timestampHistory"), idx))
                )

            # Aplicar o filtro de timestamps
            df_treino_final = df_treino_final \
                .withColumn(
                    "timestampHistory_filtered",
                    array(*timestamp_expressions)
                )

            # Registrar função temporária para filtrar nulos
            df_treino_final.createOrReplaceTempView("temp_view")

            # SQL para filtrar nulos
            sql_query = """
            SELECT 
                idUsuario,
                userType,
                historico,
                filter(timestampHistory_filtered, x -> x is not null) as timestampHistory,
                numberOfClicksHistory,
                timeOnPageHistory,
                scrollPercentageHistory,
                pageVisitsCountHistory
            FROM temp_view
            """

            df_treino_final = self.spark.sql(sql_query)

            # Log de verificação dos timestamps
            logger.info("Verificando timestamps processados:")

            # Análise dos timestamps de forma segura
            logger.info("Analisando distribuição dos timestamps:")
            timestamp_stats = df_treino_final.select(
                size(col("timestampHistory")).alias("tamanho_timestamp"),
                when(size(col("timestampHistory")) > 0, 
                    element_at(col("timestampHistory"), 1)
                ).otherwise(lit(None)).alias("primeiro_timestamp"),
                when(size(col("timestampHistory")) > 0,
                    element_at(
                        col("timestampHistory"), 
                        size(col("timestampHistory"))
                    )
                ).otherwise(lit(None)).alias("ultimo_timestamp")
            )

            # Mostrar estatísticas de forma segura
            try:
                timestamp_stats.show(truncate=False)
            except Exception as e:
                logger.warning(f"Erro ao mostrar estatísticas de timestamps: {str(e)}")
                # Alternativa mais simples para mostrar as estatísticas
                df_treino_final.select(
                    size(col("timestampHistory")).alias("tamanho_timestamp")
                ).show(5, truncate=False)

            # Estatísticas adicionais de forma segura
            try:
                logger.info("Estatísticas dos timestamps:")
                df_treino_final.agg(
                    avg(size(col("timestampHistory"))).alias("media_timestamps"),
                    min(size(col("timestampHistory"))).alias("min_timestamps"),
                    max(size(col("timestampHistory"))).alias("max_timestamps")
                ).show()
            except Exception as e:
                logger.warning(f"Erro ao calcular estatísticas adicionais: {str(e)}")

            # Calcular estatísticas finais
            n_registros_final = df_treino_final.filter(size("historico") > 0).count()
            logger.info(f"Registros finais com histórico válido: {n_registros_final}")

            # Verificar timestamps válidos
            n_timestamps_validos = df_treino_final.filter(
                size(col("timestampHistory")) > 0
            ).count()
            logger.info(f"Registros com timestamps válidos: {n_timestamps_validos}")

            logger.info("Range de datas após processamento:")
            df_itens_processado.select(
                min("DataPublicacao").alias("data_min"),
                max("DataPublicacao").alias("data_max")
            ).show()

            # Salvar dados processados
            try:
                logger.info("\nSalvando dados processados...")
                caminho_processados = "dados/processados"
                os.makedirs(caminho_processados, exist_ok=True)

                # Salvar dados de treino
                caminho_treino = f"{caminho_processados}/dados_treino_processados"
                logger.info(f"Salvando dados de treino em: {caminho_treino}")
                
                df_treino_final.write \
                    .mode("overwrite") \
                    .option("compression", "snappy") \
                    .parquet(f"{caminho_treino}.parquet")
                
                # Salvar dados de itens
                caminho_itens = f"{caminho_processados}/dados_itens_processados"
                logger.info(f"Salvando dados de itens em: {caminho_itens}")
                
                df_itens_processado.write \
                    .mode("overwrite") \
                    .option("compression", "snappy") \
                    .parquet(f"{caminho_itens}.parquet")

                # Salvar metadados
                metadados = {
                    "data_processamento": datetime.now().isoformat(),
                    "estatisticas_treino": {
                        "registros_iniciais": n_registros_inicial,
                        "registros_finais": n_registros_final,
                        "registros_com_historico_valido": n_registros_final,
                        "timestamps_validos": n_timestamps_validos
                    },
                    "estatisticas_itens": {
                        "total_itens": df_itens_processado.count()
                    },
                    "parametros_processamento": {
                        "min_year": min_year,
                        "max_year": max_year,
                        "max_timestamps": max_timestamps
                    },
                    "colunas_treino": df_treino_final.columns,
                    "colunas_itens": df_itens_processado.columns
                }

                caminho_metadados = f"{caminho_processados}/metadados.json"
                with open(caminho_metadados, 'w') as f:
                    json.dump(metadados, f, indent=4)

                logger.info("\nDados processados salvos com sucesso!")
                logger.info(f"- Treino: {caminho_treino}.parquet")
                logger.info(f"- Itens: {caminho_itens}.parquet")
                logger.info(f"- Metadados: {caminho_metadados}")

            except Exception as e:
                logger.error(f"Erro ao salvar dados processados: {str(e)}")
                logger.warning("Continuando sem salvar os dados processados...")
            
            return df_treino_final, df_itens_processado
                
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            raise
        finally:
            for df in dfs_to_unpersist:
                if df and df.is_cached:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {str(e)}")

    def carregar_dados_processados(self):
        """
        Carrega os dados processados anteriormente.
        
        Returns:
            tuple: (DataFrame de treino, DataFrame de itens)
        """
        try:
            caminho_processados = "dados/processados"
            
            if not os.path.exists(caminho_processados):
                raise FileNotFoundError("Diretório de dados processados não encontrado")
            
            logger.info("Carregando dados processados...")
            
            # Carregar dados de treino
            caminho_treino = f"{caminho_processados}/dados_treino_processados.parquet"
            df_treino = self.spark.read.parquet(caminho_treino)
            logger.info(f"Dados de treino carregados: {df_treino.count()} registros")
            
            # Carregar dados de itens
            caminho_itens = f"{caminho_processados}/dados_itens_processados.parquet"
            df_itens = self.spark.read.parquet(caminho_itens)
            logger.info(f"Dados de itens carregados: {df_itens.count()} registros")
            
            # Verificar metadados
            caminho_metadados = f"{caminho_processados}/metadados.json"
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as f:
                    metadados = json.load(f)
                logger.info("\nMetadados do processamento:")
                logger.info(f"Data do processamento: {metadados['data_processamento']}")
                logger.info(f"Número de registros original:")
                logger.info(f"- Treino: {metadados['n_registros_treino']}")
                logger.info(f"- Itens: {metadados['n_registros_itens']}")
            
            return df_treino, df_itens
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados processados: {str(e)}")
            raise


    def validar_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> bool:
        """
        Valida a qualidade dos dados processados.
        """
        logger.info("Validando dados processados")
        self._verificar_spark_ativo()
        
        dfs_to_unpersist = []
        
        try:
            # Cache para operações de validação
            df_treino.persist()
            df_itens.persist()
            dfs_to_unpersist.extend([df_treino, df_itens])
            
            # Verificar valores nulos em campos críticos
            logger.info("Verificando valores nulos em campos críticos")
            campos_criticos_treino = ["idUsuario", "historico"]
            campos_criticos_itens = ["page", "conteudo_texto", "DataPublicacao"]

            # Validar dados de treino
            nulos_treino = {
                campo: df_treino.filter(col(campo).isNull()).count() 
                for campo in campos_criticos_treino
            }
            
            # Validar dados de itens
            nulos_itens = {
                campo: df_itens.filter(col(campo).isNull()).count() 
                for campo in campos_criticos_itens
            }

            tem_nulos = False
            
            # Verificar nulos nos dados de treino
            for campo, count in nulos_treino.items():
                if count > 0:
                    logger.warning(f"Campo {campo} tem {count} valores nulos")
                    tem_nulos = True
            
            # Verificar nulos nos dados de itens
            for campo, count in nulos_itens.items():
                if count > 0:
                    logger.warning(f"Campo {campo} tem {count} valores nulos")
                    tem_nulos = True
            
            # Verificar tamanhos dos arrays no histórico
            logger.info("Verificando consistência dos arrays de histórico")
            tamanhos_diferentes = df_treino.filter(
                ~(size("historico") == size("timestampHistory")) |
                ~(size("historico") == size("numberOfClicksHistory")) |
                ~(size("historico") == size("timeOnPageHistory")) |
                ~(size("historico") == size("scrollPercentageHistory")) |
                ~(size("historico") == size("pageVisitsCountHistory"))
            ).count()

            if tamanhos_diferentes > 0:
                logger.warning(f"Encontrados {tamanhos_diferentes} registros com arrays de tamanhos diferentes")
                tem_nulos = True
            
            # Verificar consistência entre histórico e itens
            logger.info("Verificando consistência entre histórico e itens")
            historico_items = df_treino.select(
                explode("historico").alias("item")
            ).distinct().persist()
            dfs_to_unpersist.append(historico_items)
            
            itens_faltantes = historico_items.join(
                df_itens,
                historico_items.item == df_itens.page,
                "left_anti"
            ).persist()
            dfs_to_unpersist.append(itens_faltantes)
            
            n_faltantes = itens_faltantes.count()
            if n_faltantes > 0:
                logger.warning(
                    f"Existem {n_faltantes} itens no histórico não encontrados nos dados de itens"
                )
                itens_faltantes.show(5, truncate=False)
                tem_nulos = True
            
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

    def mostrar_info_dados(self, df_treino: DataFrame, df_itens: DataFrame) -> None:
        """
        Mostra informações detalhadas sobre os dados processados.
        """
        try:
            self._verificar_spark_ativo()
            
            # Informações gerais
            n_registros = df_treino.count()
            n_usuarios = df_treino.select("idUsuario").distinct().count()
            
            logger.info("\nInformações dos dados de treino:")
            logger.info(f"Número de registros: {n_registros}")
            logger.info(f"Número de usuários únicos: {n_usuarios}")
            
            # Estatísticas do histórico
            df_treino.select(
                avg(size("historico")).alias("media_itens"),
                min(size("historico")).alias("min_itens"),
                max(size("historico")).alias("max_itens")
            ).show()
            
            # Informações dos itens
            n_itens = df_itens.count()
            logger.info(f"\nNúmero de itens: {n_itens}")
            
            # Período dos dados
            logger.info("\nPeríodo dos dados:")
            df_itens.select(
                min("DataPublicacao").alias("primeira_publicacao"),
                max("DataPublicacao").alias("ultima_publicacao")
            ).show()
            
            # Mostrar exemplos dos dados
            logger.info("\nExemplos dos dados de treino:")
            df_treino.show(3, truncate=False)
            
            logger.info("\nExemplos dos dados de itens:")
            df_itens.show(3, truncate=False)
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações: {str(e)}")
            raise

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