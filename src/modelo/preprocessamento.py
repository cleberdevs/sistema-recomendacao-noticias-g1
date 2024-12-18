import pandas as pd
import numpy as np
from datetime import datetime

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None

    def processar_dados_treino(self, arquivos_treino, arquivos_itens):
        # Carregar dados de treino
        dfs_treino = []
        for arquivo in arquivos_treino:
            df = pd.read_csv(arquivo)
            dfs_treino.append(df)
        dados_treino = pd.concat(dfs_treino, ignore_index=True)
        
        # Carregar dados dos itens
        dfs_itens = []
        for arquivo in arquivos_itens:
            df = pd.read_csv(arquivo)
            dfs_itens.append(df)
        dados_itens = pd.concat(dfs_itens, ignore_index=True)
        
        # Processar histórico
        dados_treino['historico'] = dados_treino['historico'].apply(eval)
        dados_treino['historicoTimestamp'] = dados_treino['historicoTimestamp'].apply(eval)
        
        # Processar timestamps
        dados_itens['DataPublicacao'] = pd.to_datetime(dados_itens['DataPublicacao'])
        
        return dados_treino, dados_itens

    def preparar_features_texto(self, dados_itens):
        dados_itens['conteudo_texto'] = dados_itens['Titulo'] + ' ' + dados_itens['Corpo']
        return dados_itens

    def calcular_metricas_preprocessamento(self, dados_treino, dados_itens):
        return {
            "num_usuarios": dados_treino['idUsuario'].nunique(),
            "num_itens": len(dados_itens),
            "media_interacoes_por_usuario": dados_treino['historico'].apply(len).mean()
        }
