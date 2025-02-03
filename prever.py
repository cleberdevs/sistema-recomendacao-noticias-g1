import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# Adicionar diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.config.logging_config import configurar_logging
from src.config.spark_config import configurar_ambiente_spark

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def carregar_dados_brutos(spark):
    """
    Carrega os dados brutos para obter IDs originais.
    
    Returns:
        DataFrame: Dados de treino brutos
    """
    try:
        preprocessador = PreProcessadorDadosSpark(spark)
        
        # Carregar arquivo de treino
        arquivos_treino = [f for f in os.listdir('dados/brutos') 
                          if f.startswith('treino_parte') and f.endswith('.csv')]
        
        caminho_treino = [os.path.join('dados/brutos', f) for f in arquivos_treino]
        
        # Ler dados brutos
        df_treino = spark.read.csv(
            caminho_treino,
            header=True
        )
        
        return df_treino
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados brutos: {str(e)}")
        raise

def carregar_modelo():
    """
    Carrega o modelo treinado.
    """
    try:
        caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
        logger.info(f"Carregando modelo de {caminho_modelo}")
        modelo = RecomendadorHibrido.carregar_modelo(caminho_modelo)
        logger.info("Modelo carregado com sucesso")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

def mostrar_usuarios_disponiveis(spark, modelo):
    """
    Mostra os IDs originais dos usuários disponíveis.
    """
    try:
        # Carregar dados brutos
        df_treino = carregar_dados_brutos(spark)
        
        # Pegar IDs únicos
        usuarios = df_treino.select("userId").distinct().limit(10).collect()
        
        logger.info(f"\nTotal de usuários no modelo: {len(modelo.itens_usuario)}")
        logger.info("\nPrimeiros 10 IDs de usuários:")
        for i, row in enumerate(usuarios, 1):
            usuario_id = row['userId']
            logger.info(f"{i}. ID: {usuario_id}")
            
        return [row['userId'] for row in usuarios]
        
    except Exception as e:
        logger.error(f"Erro ao mostrar usuários: {str(e)}")
        raise

def mostrar_detalhes_usuario(modelo, usuario_id):
    """
    Mostra detalhes do histórico do usuário.
    """
    try:
        if usuario_id not in modelo.itens_usuario:
            logger.info(f"Usuário {usuario_id} não encontrado")
            return
            
        historico = modelo.itens_usuario[usuario_id]
        logger.info(f"\nDetalhes do usuário {usuario_id}:")
        logger.info(f"Tamanho do histórico: {len(historico)}")
        
        logger.info("\nÚltimos 5 itens do histórico:")
        for idx in list(historico)[-5:]:
            url = modelo.index_to_item_id[idx]
            logger.info(f"- URL: {url}")
            
    except Exception as e:
        logger.error(f"Erro ao mostrar detalhes do usuário: {str(e)}")

def fazer_previsoes(modelo, usuario_id, n_recomendacoes=5):
    """
    Faz previsões para um usuário específico.
    """
    try:
        logger.info(f"Fazendo previsões para usuário {usuario_id}")
        
        # Verificar se usuário existe no modelo
        if usuario_id not in modelo.itens_usuario:
            logger.warning(f"Usuário {usuario_id} não encontrado no conjunto de treino")
            return []
        
        # Obter histórico do usuário
        historico = modelo.itens_usuario[usuario_id]
        logger.info(f"Usuário tem {len(historico)} itens no histórico")
        
        # Obter todos os itens disponíveis
        todos_itens = set(modelo.features_item.keys())
        
        # Remover itens que o usuário já interagiu
        itens_candidatos = list(todos_itens - historico)
        
        # Preparar dados para previsão
        n_candidatos = len(itens_candidatos)
        X_usuario = np.array([usuario_id] * n_candidatos)
        X_item = np.array(itens_candidatos)
        X_conteudo = np.array([modelo.features_item[idx]['vetor_conteudo'] 
                              for idx in itens_candidatos])
        
        # Fazer previsões
        logger.info("Realizando previsões")
        previsoes = modelo.modelo.predict(
            [X_usuario, X_item, X_conteudo],
            batch_size=64,
            verbose=0
        )
        
        # Ordenar itens por probabilidade
        indices_ordenados = np.argsort(previsoes.flatten())[::-1][:n_recomendacoes]
        itens_recomendados = [itens_candidatos[i] for i in indices_ordenados]
        
        # Converter índices para URLs
        urls_recomendadas = [modelo.index_to_item_id[idx] for idx in itens_recomendados]
        
        # Log das recomendações
        logger.info("\nRecomendações geradas:")
        for i, (url, prob) in enumerate(zip(urls_recomendadas, previsoes[indices_ordenados]), 1):
            logger.info(f"\n{i}. Recomendação:")
            logger.info(f"   URL: {url}")
            logger.info(f"   Probabilidade: {prob[0]:.4f}")
        
        return urls_recomendadas
        
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise

def main():
    """
    Função principal para testar previsões.
    """
    spark = None
    try:
        # Configurar Spark
        spark = SparkSession.builder \
            .appName("PrevisaoRecomendador") \
            .getOrCreate()
            
        # Carregar modelo
        modelo = carregar_modelo()
        
        # Mostrar usuários disponíveis
        usuarios = mostrar_usuarios_disponiveis(spark, modelo)
        
        # Permitir escolha do usuário
        while True:
            try:
                escolha = input("\nEscolha o número do usuário (1-10) ou 'q' para sair: ")
                
                if escolha.lower() == 'q':
                    break
                    
                idx = int(escolha) - 1
                if 0 <= idx < len(usuarios):
                    usuario_id = usuarios[idx]
                    
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Análise do usuário selecionado:")
                    logger.info(f"ID do usuário: {usuario_id}")
                    
                    # Mostrar detalhes do usuário
                    mostrar_detalhes_usuario(modelo, usuario_id)
                    
                    # Fazer previsões
                    recomendacoes = fazer_previsoes(modelo, usuario_id, n_recomendacoes=5)
                    
                    if not recomendacoes:
                        logger.warning("Nenhuma recomendação gerada")
                        continue
                        
                    # Perguntar se quer continuar
                    continuar = input("\nDeseja ver outro usuário? (s/n): ")
                    if continuar.lower() != 's':
                        break
                else:
                    logger.warning("Número de usuário inválido. Escolha entre 1 e 10.")
                    
            except ValueError:
                logger.warning("Por favor, digite um número válido.")
                
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        raise
    finally:
        logger.info("\nTeste de previsões concluído")
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()