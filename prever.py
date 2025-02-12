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

def mostrar_usuarios_disponiveis(spark, modelo, limite=None, pagina=1, usuarios_por_pagina=20):
    """
    Mostra os IDs originais dos usuários disponíveis com paginação.
    """
    try:
        # Mostrar informações sobre os IDs disponíveis no modelo
        usuarios_modelo = sorted(modelo.usuario_id_to_index.keys())
        
        logger.info(f"\nTotal de usuários no modelo: {len(usuarios_modelo)}")
        logger.info("\nExemplos de IDs válidos:")
        for i, usuario_id in enumerate(usuarios_modelo[:5], 1):
            n_historico = len(modelo.itens_usuario.get(usuario_id, []))
            logger.info(f"{i}. ID: {usuario_id} (Itens no histórico: {n_historico})")
        
        return usuarios_modelo
        
    except Exception as e:
        logger.error(f"Erro ao mostrar usuários: {str(e)}")
        raise


def mostrar_detalhes_usuario(modelo, usuario_id):
    """
    Mostra detalhes do histórico do usuário.
    """
    try:
        usuario_id = str(usuario_id)
        if usuario_id not in modelo.usuario_id_to_index:
            logger.info(f"Usuário {usuario_id} não encontrado")
            return
            
        historico = modelo.itens_usuario.get(usuario_id, set())
        logger.info(f"\nDetalhes do usuário {usuario_id}:")
        logger.info(f"Tamanho do histórico: {len(historico)}")
        
        logger.info("\nÚltimos 5 itens do histórico:")
        for idx in list(historico)[-5:]:
            if idx in modelo.index_to_item_id:
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
        
        # Converter usuario_id para string para garantir consistência
        usuario_id = str(usuario_id)
        
        # Verificar se usuário existe no modelo
        if usuario_id not in modelo.usuario_id_to_index:
            logger.warning(f"Usuário {usuario_id} não encontrado no conjunto de treino")
            return []
        
        # Obter índice numérico do usuário
        usuario_idx = modelo.usuario_id_to_index[usuario_id]
        
        # Obter histórico do usuário
        historico = modelo.itens_usuario.get(usuario_id, set())
        logger.info(f"Usuário tem {len(historico)} itens no histórico")
        
        # Obter todos os itens disponíveis
        todos_itens = set(modelo.features_item.keys())
        
        # Remover itens que o usuário já interagiu
        itens_candidatos = list(todos_itens - historico)
        
        # Preparar dados para previsão em lotes
        batch_size = 1000
        todas_previsoes = []
        
        for i in range(0, len(itens_candidatos), batch_size):
            batch_candidatos = itens_candidatos[i:i + batch_size]
            
            # Criar arrays para o batch atual
            X_usuario = np.full(len(batch_candidatos), usuario_idx, dtype=np.int32)
            X_item = np.array(batch_candidatos, dtype=np.int32)
            X_conteudo = np.array([
                modelo.features_item[idx]['vetor_conteudo'] 
                for idx in batch_candidatos
            ], dtype=np.float32)
            
            # Fazer previsões para o batch
            batch_previsoes = modelo.modelo.predict(
                [X_usuario, X_item, X_conteudo],
                batch_size=64,
                verbose=0
            )
            todas_previsoes.extend(batch_previsoes.flatten())
        
        # Converter para array numpy
        previsoes = np.array(todas_previsoes)
        
        # Ordenar itens por probabilidade
        indices_ordenados = np.argsort(previsoes)[::-1][:n_recomendacoes]
        itens_recomendados = [itens_candidatos[i] for i in indices_ordenados]
        probabilidades = previsoes[indices_ordenados]
        
        # Converter índices para URLs e criar lista de recomendações
        recomendacoes = []
        for idx, prob in zip(itens_recomendados, probabilidades):
            if idx in modelo.index_to_item_id:
                url = modelo.index_to_item_id[idx]
                recomendacoes.append({
                    'url': url,
                    'probabilidade': float(prob)
                })
                logger.info(f"\n{len(recomendacoes)}. Recomendação:")
                logger.info(f"   URL: {url}")
                logger.info(f"   Probabilidade: {prob:.4f}")
        
        return recomendacoes
        
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
        
        while True:
            # Mostrar usuários disponíveis e exemplo de IDs válidos
            usuarios_disponiveis = mostrar_usuarios_disponiveis(spark, modelo)
            
            print("\n=== MENU DE OPÇÕES ===")
            print("Digite:")
            print("- 'b' para buscar um usuário específico por ID")
            print("- 'l' para listar mais IDs válidos")
            print("- 'q' para sair")
            
            escolha = input("\nSua escolha: ").lower()
            
            if escolha == 'q':
                break
            elif escolha == 'l':
                # Mostrar mais IDs válidos
                print("\nIDs válidos disponíveis:")
                for i, usuario_id in enumerate(usuarios_disponiveis[:20], 1):
                    print(f"{i}. ID: {usuario_id}")
                continue
            elif escolha == 'b':
                print("\n=== BUSCA DE USUÁRIO ===")
                usuario_busca = input("Digite o ID do usuário: ")
                
                # Verificar se o ID existe no modelo
                if usuario_busca not in modelo.usuario_id_to_index:
                    logger.warning(f"ID {usuario_busca} não encontrado no modelo.")
                    logger.info("Use um dos IDs válidos mostrados acima.")
                    continue
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Análise do usuário buscado:")
                logger.info(f"ID do usuário: {usuario_busca}")
                
                # Mostrar detalhes do usuário
                mostrar_detalhes_usuario(modelo, usuario_busca)
                
                # Fazer previsões
                recomendacoes = fazer_previsoes(modelo, usuario_busca, n_recomendacoes=5)
                
                if not recomendacoes:
                    logger.warning("Nenhuma recomendação gerada")
                    
                input("\nPressione Enter para continuar...")
            else:
                logger.warning("Opção inválida")
                
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        raise
    finally:
        logger.info("\nTeste de previsões concluído")
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()