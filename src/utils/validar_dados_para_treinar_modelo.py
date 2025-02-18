import os
import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

def configurar_logging():
    """Configura o sistema de logging."""
    # Criar diretório de logs se não existir
    os.makedirs('logs', exist_ok=True)
    
    # Configurar formato do log
    formato_log = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Arquivo de log com data
    arquivo_log = f'logs/validacao_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(arquivo_log)
    file_handler.setFormatter(formato_log)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formato_log)
    
    # Configurar logger root
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remover handlers existentes
    logger.handlers = []
    
    # Adicionar novos handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Configurar logging
logger = configurar_logging()

# Requisitos mínimos para os dados
REQUISITOS_MINIMOS = {
    'usuarios_ativos': 1000,      # Número mínimo de usuários
    'interacoes_por_usuario': 10, # Média de interações por usuário
    'itens_por_categoria': 50,    # Mínimo de itens por categoria
    'categorias': 5,              # Número mínimo de categorias
    'cobertura_features': 0.95    # % de itens com features completas
}

def criar_spark_session():
    """Cria e configura uma sessão Spark para validação."""
    return SparkSession.builder \
        .appName("ValidacaoDados") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def verificar_conexao_spark(spark, max_tentativas=3):
    """Verifica se a conexão Spark está funcionando."""
    for tentativa in range(max_tentativas):
        try:
            test_df = spark.createDataFrame([(1,)], ["test"])
            test_df.collect()
            return True
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
            if tentativa < max_tentativas - 1:
                import time
                time.sleep(5)
    return False

def validar_schema_dados(df_treino, df_itens):
    """Valida os schemas dos DataFrames."""
    logger.info("Validando schemas dos dados...")
    
    # Schema esperado para dados de treino
    schema_treino_esperado = {
        'idUsuario': 'string',
        'historico': 'array',
        'timestampHistory': 'array',
        'numberOfClicksHistory': 'array',
        'timeOnPageHistory': 'array',
        'scrollPercentageHistory': 'array',
        'pageVisitsCountHistory': 'array'
    }
    
    # Schema esperado para dados de itens
    schema_itens_esperado = {
        'page': 'string',
        'url': 'string',
        'DataPublicacao': 'timestamp',
        'conteudo_texto': 'string'
    }
    
    # Verificar colunas do treino
    colunas_treino = {col: str(tipo) for col, tipo in df_treino.dtypes}
    colunas_faltantes_treino = set(schema_treino_esperado.keys()) - set(colunas_treino.keys())
    if colunas_faltantes_treino:
        logger.error(f"Colunas faltantes nos dados de treino: {colunas_faltantes_treino}")
        return False
        
    # Verificar colunas dos itens
    colunas_itens = {col: str(tipo) for col, tipo in df_itens.dtypes}
    colunas_faltantes_itens = set(schema_itens_esperado.keys()) - set(colunas_itens.keys())
    if colunas_faltantes_itens:
        logger.error(f"Colunas faltantes nos dados de itens: {colunas_faltantes_itens}")
        return False
    
    # Log dos schemas atuais
    logger.info("\nSchema atual dos dados de treino:")
    for col, tipo in colunas_treino.items():
        logger.info(f"- {col}: {tipo}")
        
    logger.info("\nSchema atual dos dados de itens:")
    for col, tipo in colunas_itens.items():
        logger.info(f"- {col}: {tipo}")
    
    return True

def validar_integridade_dados(df_treino, df_itens):
    """Valida a integridade dos dados."""
    logger.info("Validando integridade dos dados...")
    
    try:
        # Verificar valores nulos
        nulos_treino = df_treino.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df_treino.columns
        ]).first()
        
        nulos_itens = df_itens.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df_itens.columns
        ]).first()
        
        # Log de valores nulos
        logger.info("\nValores nulos nos dados de treino:")
        tem_nulos_treino = False
        for col in df_treino.columns:
            n_nulos = getattr(nulos_treino, col)
            if n_nulos > 0:
                tem_nulos_treino = True
                logger.warning(f"Coluna {col}: {n_nulos} valores nulos")
                
        logger.info("\nValores nulos nos dados de itens:")
        tem_nulos_itens = False
        for col in df_itens.columns:
            n_nulos = getattr(nulos_itens, col)
            if n_nulos > 0:
                tem_nulos_itens = True
                logger.warning(f"Coluna {col}: {n_nulos} valores nulos")
        
        # Verificar tamanhos dos arrays no histórico
        problemas_arrays = df_treino.filter(
            ~(F.size("historico") == F.size("timestampHistory")) |
            ~(F.size("historico") == F.size("numberOfClicksHistory")) |
            ~(F.size("historico") == F.size("timeOnPageHistory")) |
            ~(F.size("historico") == F.size("scrollPercentageHistory")) |
            ~(F.size("historico") == F.size("pageVisitsCountHistory"))
        ).count()
        
        if problemas_arrays > 0:
            logger.error(f"Encontrados {problemas_arrays} registros com arrays de tamanhos inconsistentes")
            return False
            
        # Verificar valores duplicados
        duplicados_treino = df_treino.groupBy("idUsuario").count().filter("count > 1").count()
        duplicados_itens = df_itens.groupBy("page").count().filter("count > 1").count()
        
        if duplicados_treino > 0:
            logger.warning(f"Encontrados {duplicados_treino} usuários duplicados")
        if duplicados_itens > 0:
            logger.warning(f"Encontrados {duplicados_itens} itens duplicados")
        
        # Validar timestamps
        datas_invalidas = df_itens.filter(
            (F.year("DataPublicacao") < 1970) | 
            (F.year("DataPublicacao") > 2030)
        ).count()
        
        if datas_invalidas > 0:
            logger.warning(f"Encontradas {datas_invalidas} datas de publicação inválidas")
        
        return not (tem_nulos_treino or tem_nulos_itens or problemas_arrays > 0)
        
    except Exception as e:
        logger.error(f"Erro na validação de integridade: {str(e)}")
        return False

def validar_consistencia_dados(df_treino, df_itens):
    """Valida a consistência entre dados de treino e itens."""
    logger.info("Validando consistência entre dados de treino e itens...")
    
    try:
        # Extrair URLs únicas do histórico
        urls_historico = df_treino.select(F.explode("historico").alias("url")).distinct()
        
        # Verificar URLs não encontradas nos itens
        urls_nao_encontradas = urls_historico.join(
            df_itens,
            urls_historico.url == df_itens.page,
            "left_anti"
        )
        
        n_urls_nao_encontradas = urls_nao_encontradas.count()
        if n_urls_nao_encontradas > 0:
            logger.warning(f"{n_urls_nao_encontradas} URLs do histórico não encontradas nos itens")
            logger.warning("Exemplos de URLs não encontradas:")
            urls_nao_encontradas.show(5, truncate=False)
            
        # Calcular estatísticas
        total_urls_historico = urls_historico.count()
        total_urls_itens = df_itens.select("page").distinct().count()
        
        logger.info(f"\nTotal de URLs únicas no histórico: {total_urls_historico}")
        logger.info(f"Total de URLs únicas nos itens: {total_urls_itens}")
        logger.info(f"URLs não encontradas: {n_urls_nao_encontradas}")
        
        # Calcular cobertura
        cobertura = (total_urls_historico - n_urls_nao_encontradas) / total_urls_historico
        logger.info(f"Cobertura de URLs: {cobertura:.2%}")
        
        return cobertura >= 0.5  # Requer pelo menos 50% de cobertura
        
    except Exception as e:
        logger.error(f"Erro na validação de consistência: {str(e)}")
        return False

def validar_distribuicao_dados(df_treino, df_itens):
    """Valida a distribuição dos dados."""
    logger.info("Analisando distribuição dos dados...")
    
    try:
        # Estatísticas do histórico
        stats_historico = df_treino.select(
            F.avg(F.size("historico")).alias("media_itens"),
            F.min(F.size("historico")).alias("min_itens"),
            F.max(F.size("historico")).alias("max_itens"),
            F.expr("percentile_approx(size(historico), 0.5)").alias("mediana_itens")
        ).first()
        
        logger.info("\nEstatísticas do histórico:")
        logger.info(f"Média de itens por usuário: {stats_historico.media_itens:.2f}")
        logger.info(f"Mínimo de itens: {stats_historico.min_itens}")
        logger.info(f"Máximo de itens: {stats_historico.max_itens}")
        logger.info(f"Mediana de itens: {stats_historico.mediana_itens}")
        
        # Distribuição temporal dos itens
        logger.info("\nDistribuição temporal dos itens:")
        df_itens.select(
            F.year("DataPublicacao").alias("ano")
        ).groupBy("ano").count().orderBy("ano").show()
        
        # Distribuição do tamanho dos textos
        stats_texto = df_itens.select(
            F.avg(F.length("conteudo_texto")).alias("media_caracteres"),
            F.min(F.length("conteudo_texto")).alias("min_caracteres"),
            F.max(F.length("conteudo_texto")).alias("max_caracteres")
        ).first()
        
        logger.info("\nEstatísticas dos textos:")
        logger.info(f"Média de caracteres: {stats_texto.media_caracteres:.2f}")
        logger.info(f"Mínimo de caracteres: {stats_texto.min_caracteres}")
        logger.info(f"Máximo de caracteres: {stats_texto.max_caracteres}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro na análise de distribuição: {str(e)}")
        return False

def validar_requisitos_minimos(df_treino, df_itens):
    """Valida se os dados atendem aos requisitos mínimos estabelecidos."""
    logger.info("Validando requisitos mínimos dos dados...")
    
    resultados = {}
    try:
        # 1. Número de usuários ativos
        n_usuarios = df_treino.select("idUsuario").distinct().count()
        resultados['usuarios_ativos'] = {
            'valor': n_usuarios,
            'atende': n_usuarios >= REQUISITOS_MINIMOS['usuarios_ativos']
        }
        
        # 2. Média de interações por usuário
        media_interacoes = df_treino.select(
            F.avg(F.size("historico")).alias("media")
        ).first().media
        
        resultados['interacoes_por_usuario'] = {
            'valor': media_interacoes,
            'atende': media_interacoes >= REQUISITOS_MINIMOS['interacoes_por_usuario']
        }
        
        # 3. Cobertura de features
        total_itens = df_itens.count()
        itens_com_features = df_itens.filter(
            F.col('conteudo_texto').isNotNull() & 
            (F.length('conteudo_texto') > 0)
        ).count()
        
        cobertura = itens_com_features / total_itens if total_itens > 0 else 0
        resultados['cobertura_features'] = {
            'valor': cobertura,
            'atende': cobertura >= REQUISITOS_MINIMOS['cobertura_features']
        }
        
        # Log dos resultados
        logger.info("\n=== Validação dos Requisitos Mínimos ===")
        for requisito, dados in resultados.items():
            status = "✓ ATENDE" if dados['atende'] else "✗ NÃO ATENDE"
            valor = dados['valor']
            minimo = REQUISITOS_MINIMOS[requisito]
            
            # Formatação especial para diferentes tipos de métricas
            if requisito == 'cobertura_features':
                valor_fmt = f"{valor:.2%}"
                minimo_fmt = f"{minimo:.2%}"
            else:
                valor_fmt = f"{valor:,.2f}" if isinstance(valor, float) else f"{valor:,}"
                minimo_fmt = f"{minimo:,}"
            
            logger.info(f"{requisito}:")
            logger.info(f"  Valor atual: {valor_fmt}")
            logger.info(f"  Mínimo requerido: {minimo_fmt}")
            logger.info(f"  Status: {status}\n")
        
        # Verificar se todos os requisitos são atendidos
        todos_atendidos = all(resultado['atende'] for resultado in resultados.values())
        
        if todos_atendidos:
            logger.info("✓ Todos os requisitos mínimos são atendidos")
        else:
            logger.warning("✗ Alguns requisitos mínimos não são atendidos")
        
        return todos_atendidos
        
    except Exception as e:
        logger.error(f"Erro na validação dos requisitos mínimos: {str(e)}")
        return False

def gerar_relatorio_validacao(validacoes):
    """Gera um relatório detalhado da validação."""
    try:
        logger.info("\n" + "="*50)
        logger.info("RELATÓRIO DE VALIDAÇÃO DOS DADOS")
        logger.info("="*50)
        
        for nome, resultado in validacoes:
            status = "✓ PASSOU" if resultado else "✗ FALHOU"
            logger.info(f"\n{nome}:")
            logger.info(f"Status: {status}")
            
        n_passou = sum(1 for _, resultado in validacoes if resultado)
        n_total = len(validacoes)
        
        logger.info("\n" + "="*50)
        logger.info(f"Resultado Final: {n_passou}/{n_total} validações passaram")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {str(e)}")

def main():
    """Função principal de validação."""
    spark = None
    try:
        logger.info("Iniciando validação dos dados parquet...")
        
        spark = criar_spark_session()
        
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Não foi possível estabelecer conexão com o Spark")
        
        # Caminhos dos arquivos
        caminho_treino = "dados/processados/dados_treino_processados.parquet"
        caminho_itens = "dados/processados/dados_itens_processados.parquet"
        
        # Verificar existência dos arquivos
        if not os.path.exists(caminho_treino) or not os.path.exists(caminho_itens):
            raise FileNotFoundError("Arquivos parquet não encontrados")
        
        # Carregar dados
        logger.info("Carregando dados...")
        df_treino = spark.read.parquet(caminho_treino)
        df_itens = spark.read.parquet(caminho_itens)
        
        # Mostrar informações básicas
        logger.info(f"\nRegistros de treino: {df_treino.count():,}")
        logger.info(f"Registros de itens: {df_itens.count():,}")
        
        # Persistir DataFrames para melhor performance
        df_treino.persist()
        df_itens.persist()
        
        # Executar validações
        validacoes = [
            ("Schema dos Dados", validar_schema_dados(df_treino, df_itens)),
            ("Integridade dos Dados", validar_integridade_dados(df_treino, df_itens)),
            ("Consistência entre Treino e Itens", validar_consistencia_dados(df_treino, df_itens)),
            ("Distribuição dos Dados", validar_distribuicao_dados(df_treino, df_itens)),
            ("Requisitos Mínimos", validar_requisitos_minimos(df_treino, df_itens))
        ]
        
        # Gerar relatório
        gerar_relatorio_validacao(validacoes)
        
        # Verificar resultado final
        todas_validacoes_ok = all(resultado for _, resultado in validacoes)
        
        if todas_validacoes_ok:
            logger.info("\n✓ Dados adequados para treinamento")
            return True
        else:
            logger.warning("\n✗ Dados precisam ser corrigidos antes do treinamento")
            return False
            
    except Exception as e:
        logger.error(f"Erro durante validação: {str(e)}")
        return False
    finally:
        if spark:
            # Limpar cache
            if 'df_treino' in locals():
                df_treino.unpersist()
            if 'df_itens' in locals():
                df_itens.unpersist()
            # Parar sessão Spark
            spark.stop()

if __name__ == "__main__":
    main()