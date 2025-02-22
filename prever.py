import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

# Adicionar diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.modelo.recomendador import RecomendadorHibrido
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def calcular_recencia(timestamp_item, timestamp_atual):
    """
    Calcula score de recência baseado na idade do item.
    
    Returns:
        float: Score entre 0 e 1, onde:
        1.0 = muito recente (hoje)
        0.8 = última semana
        0.6 = último mês
        0.4 = últimos 3 meses
        0.2 = últimos 6 meses
        0.1 = mais antigo
    """
    idade_dias = (timestamp_atual - timestamp_item) / 86400  # converter para dias
    
    if idade_dias <= 1:  # Hoje
        return 1.0
    elif idade_dias <= 7:  # Última semana
        return 0.8
    elif idade_dias <= 30:  # Último mês
        return 0.6
    elif idade_dias <= 90:  # Últimos 3 meses
        return 0.4
    elif idade_dias <= 180:  # Últimos 6 meses
        return 0.2
    else:  # Mais antigo
        return 0.1

def gerar_recomendacoes_cold_start(modelo, timestamps_items, popularidade_items, k=5):
    """
    Gera recomendações para usuários novos baseado em popularidade e recência.
    """
    try:
        logger.info("Iniciando geração de recomendações cold start")
        
        # Timestamp atual para cálculo de recência
        timestamp_atual = datetime.now().timestamp()
        
        # Calcular scores combinados para cada item
        items_scores = []
        for item_idx in modelo.features_item.keys():
            if item_idx in modelo.index_to_item_id:
                url = modelo.index_to_item_id[item_idx]
                
                # Pular item se não tiver timestamp ou popularidade
                if url not in timestamps_items or url not in popularidade_items:
                    continue
                
                # Obter popularidade do arquivo de treino
                popularidade = popularidade_items[url]
                
                # Calcular recência
                timestamp_item = timestamps_items[url]
                recencia = calcular_recencia(timestamp_item, timestamp_atual)
                
                # Combinar scores (70% popularidade, 30% recência)
                score_final = (0.7 * popularidade) + (0.3 * recencia)
                
                items_scores.append({
                    'url': url,
                    'score': float(score_final),
                    'popularidade': float(popularidade),
                    'recencia': float(recencia),
                    'data_publicacao': datetime.fromtimestamp(timestamp_item).strftime('%Y-%m-%d'),
                    'tipo': 'cold_start'
                })
        
        # Ordenar por score e selecionar top-k
        items_scores.sort(key=lambda x: x['score'], reverse=True)
        recomendacoes = items_scores[:k]
        
        logger.info(f"Geradas {len(recomendacoes)} recomendações cold start")
        for i, rec in enumerate(recomendacoes[:3], 1):
            logger.info(
                f"Top {i}: score={rec['score']:.4f}, "
                f"pop={rec['popularidade']:.4f}, "
                f"rec={rec['recencia']:.4f}, "
                f"data={rec['data_publicacao']}"
            )
        
        return recomendacoes
        
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações cold start: {str(e)}")
        return []

def gerar_recomendacoes_hibridas(modelo, usuario_id, timestamps_items, popularidade_items, k=5):
    """
    Gera recomendações híbridas combinando modelo neural, popularidade e recência.
    """
    try:
        logger.info(f"Gerando recomendações híbridas para usuário {usuario_id}")
        
        # Preparar dados para o modelo
        usuario_idx = modelo.usuario_id_to_index[usuario_id]
        historico = modelo.itens_usuario.get(usuario_id, set())
        candidatos = [idx for idx in modelo.features_item.keys() if idx not in historico]
        
        if not candidatos:
            logger.warning("Nenhum item candidato disponível")
            return []
        
        # Calcular scores do modelo neural
        X_usuario = np.array([usuario_idx] * len(candidatos))
        X_item = np.array(candidatos)
        X_conteudo = np.array([
            modelo.features_item[idx]['vetor_conteudo'] 
            for idx in candidatos
        ])
        
        # Fazer previsões em lotes
        batch_size = 1000
        scores_modelo = []
        
        for i in range(0, len(candidatos), batch_size):
            batch_end = min(i + batch_size, len(candidatos))
            batch_scores = modelo.modelo.predict(
                [
                    X_usuario[i:batch_end],
                    X_item[i:batch_end],
                    X_conteudo[i:batch_end]
                ],
                verbose=0
            )
            scores_modelo.extend(batch_scores.flatten())
        
        # Timestamp atual para cálculo de recência
        timestamp_atual = datetime.now().timestamp()
        
        # Combinar scores para cada item candidato
        recomendacoes = []
        for i, item_idx in enumerate(candidatos):
            if item_idx in modelo.index_to_item_id:
                url = modelo.index_to_item_id[item_idx]
                
                # Pular item se não tiver timestamp ou popularidade
                if url not in timestamps_items or url not in popularidade_items:
                    continue
                
                # Score do modelo neural
                score_modelo = float(scores_modelo[i])
                
                # Score de popularidade do arquivo de treino
                popularidade = popularidade_items[url]
                
                # Score de recência
                timestamp_item = timestamps_items[url]
                recencia = calcular_recencia(timestamp_item, timestamp_atual)
                
                # Combinar scores (60% modelo, 25% popularidade, 15% recência)
                score_final = (0.60 * score_modelo) + \
                            (0.25 * popularidade) + \
                            (0.15 * recencia)
                
                recomendacoes.append({
                    'url': url,
                    'score': float(score_final),
                    'score_modelo': score_modelo,
                    'popularidade': float(popularidade),
                    'recencia': float(recencia),
                    'data_publicacao': datetime.fromtimestamp(timestamp_item).strftime('%Y-%m-%d'),
                    'tipo': 'modelo'
                })
        
        # Ordenar por score final e retornar top-k
        recomendacoes.sort(key=lambda x: x['score'], reverse=True)
        recomendacoes = recomendacoes[:k]
        
        logger.info(f"Geradas {len(recomendacoes)} recomendações híbridas")
        for i, rec in enumerate(recomendacoes[:3], 1):
            logger.info(
                f"Top {i}: score={rec['score']:.4f}, "
                f"modelo={rec['score_modelo']:.4f}, "
                f"pop={rec['popularidade']:.4f}, "
                f"rec={rec['recencia']:.4f}, "
                f"data={rec['data_publicacao']}"
            )
        
        return recomendacoes
        
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações híbridas: {str(e)}")
        return []

def fazer_previsoes(modelo, usuario_id, timestamps_items, popularidade_items, n_recomendacoes=5):
    """
    Faz previsões para um usuário específico, considerando cold start e modelo híbrido.
    """
    try:
        logger.info(f"Fazendo previsões para usuário {usuario_id}")
        
        # Verificar se é um caso de cold start
        is_cold_start = usuario_id.startswith('novo_usuario_')

        if is_cold_start:
            logger.info(f"Usuário {usuario_id} não possui histórico - aplicando cold start")
            return gerar_recomendacoes_cold_start(
                modelo, 
                timestamps_items,
                popularidade_items,
                k=n_recomendacoes
            )
        
        # Se não é cold start, usar recomendações híbridas
        logger.info(f"Gerando recomendações híbridas para usuário existente {usuario_id}")
        return gerar_recomendacoes_hibridas(
            modelo, 
            usuario_id, 
            timestamps_items,
            popularidade_items,
            k=n_recomendacoes
        )

    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        return []

if __name__ == "__main__":
    # Código para testes diretos do script
    try:
        modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
        
        # Carregar dados para teste
        spark = SparkSession.builder \
            .appName("TesteRecomendador") \
            .getOrCreate()
            
        # Carregar timestamps e popularidade
        caminho_itens = 'dados/processados/dados_itens_processados.parquet'
        caminho_treino = 'dados/processados/dados_treino_processados.parquet'
        
        # Carregar alguns dados para teste
        timestamps_teste = {}
        popularidade_teste = {}
        
        # Testar com usuário novo
        usuario_novo = "novo_usuario_1"
        recomendacoes = fazer_previsoes(
            modelo, 
            usuario_novo, 
            timestamps_teste,
            popularidade_teste,
            n_recomendacoes=5
        )
        print(f"\nRecomendações para usuário novo ({usuario_novo}):")
        for rec in recomendacoes:
            print(
                f"URL: {rec['url']}, "
                f"Score: {rec['score']:.4f}, "
                f"Pop: {rec['popularidade']:.4f}, "
                f"Data: {rec['data_publicacao']}"
            )
        
        # Testar com usuário existente
        if modelo.usuario_id_to_index:
            usuario_existente = list(modelo.usuario_id_to_index.keys())[0]
            recomendacoes = fazer_previsoes(
                modelo, 
                usuario_existente, 
                timestamps_teste,
                popularidade_teste,
                n_recomendacoes=5
            )
            print(f"\nRecomendações para usuário existente ({usuario_existente}):")
            for rec in recomendacoes:
                print(
                    f"URL: {rec['url']}, "
                    f"Score: {rec['score']:.4f}, "
                    f"Pop: {rec['popularidade']:.4f}, "
                    f"Data: {rec['data_publicacao']}"
                )
                
    except Exception as e:
        logger.error(f"Erro nos testes: {str(e)}")
        
    finally:
        if 'spark' in locals():
            spark.stop()