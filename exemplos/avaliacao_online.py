import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class AvaliacaoOnline:
    """
    Implementa um sistema de avaliação online (A/B Testing) para o recomendador.
    """
    
    def __init__(self):
        self.metricas = defaultdict(list)
        self.log_file = 'logs/avaliacao_online.jsonl'
        self.janela_avaliacao = timedelta(hours=1)
        
    def registrar_interacao(self, 
                          usuario_id: str,
                          grupo: str,  # 'A' ou 'B'
                          recomendacoes: List[str],
                          interacoes: List[str],
                          tempo_visualizacao: float,
                          clicks: int):
        """Registra uma interação do usuário com as recomendações."""
        try:
            # Calcular métricas
            precision = len(set(recomendacoes) & set(interacoes)) / len(recomendacoes)
            recall = len(set(recomendacoes) & set(interacoes)) / len(interacoes) if interacoes else 0
            
            # Registrar dados
            dados = {
                'timestamp': datetime.now().isoformat(),
                'usuario_id': usuario_id,
                'grupo': grupo,
                'recomendacoes': recomendacoes,
                'interacoes': interacoes,
                'tempo_visualizacao': tempo_visualizacao,
                'clicks': clicks,
                'precision': precision,
                'recall': recall
            }
            
            # Salvar em arquivo
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(dados) + '\n')
                
            # Atualizar métricas em memória
            self.metricas[grupo].append(dados)
            
        except Exception as e:
            logger.error(f"Erro ao registrar interação: {str(e)}")

    def calcular_metricas(self, janela_horas: int = 1) -> Dict:
        """
        Calcula métricas para cada grupo considerando uma janela de tempo.
        """
        try:
            # Ler dados do arquivo
            dados = []
            limite_tempo = datetime.now() - timedelta(hours=janela_horas)
            
            with open(self.log_file, 'r') as f:
                for linha in f:
                    registro = json.loads(linha)
                    timestamp = datetime.fromisoformat(registro['timestamp'])
                    if timestamp > limite_tempo:
                        dados.append(registro)
            
            if not dados:
                logger.warning("Sem dados para análise na janela de tempo especificada")
                return {}
            
            # Converter para DataFrame
            df = pd.DataFrame(dados)
            
            # Calcular métricas por grupo
            metricas_grupos = {}
            for grupo in df['grupo'].unique():
                df_grupo = df[df['grupo'] == grupo]
                
                metricas_grupos[grupo] = {
                    'usuarios_unicos': df_grupo['usuario_id'].nunique(),
                    'total_interacoes': len(df_grupo),
                    'precision_media': df_grupo['precision'].mean(),
                    'recall_medio': df_grupo['recall'].mean(),
                    'tempo_medio': df_grupo['tempo_visualizacao'].mean(),
                    'clicks_medio': df_grupo['clicks'].mean()
                }
            
            # Análise estatística
            if 'A' in metricas_grupos and 'B' in metricas_grupos:
                grupo_a = df[df['grupo'] == 'A']['precision']
                grupo_b = df[df['grupo'] == 'B']['precision']
                
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(grupo_a, grupo_b)
                
                metricas_grupos['analise_estatistica'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significativo': p_value < 0.05
                }
            
            return metricas_grupos
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {str(e)}")
            return {}

    def gerar_relatorio(self, janela_horas: int = 24):
        """Gera relatório detalhado da avaliação online."""
        try:
            metricas = self.calcular_metricas(janela_horas)
            
            if not metricas:
                logger.warning("Sem dados suficientes para gerar relatório")
                return
            
            # Criar relatório
            relatorio = {
                'periodo': {
                    'inicio': (datetime.now() - timedelta(hours=janela_horas)).isoformat(),
                    'fim': datetime.now().isoformat()
                },
                'metricas_por_grupo': metricas
            }
            
            # Salvar relatório
            nome_arquivo = f'logs/relatorio_ab_{datetime.now().strftime("%Y%m%d")}.json'
            with open(nome_arquivo, 'w') as f:
                json.dump(relatorio, f, indent=4)
            
            logger.info(f"Relatório gerado: {nome_arquivo}")
            
            # Mostrar resumo
            if 'analise_estatistica' in metricas:
                logger.info("\nResultados do teste A/B:")
                logger.info(f"P-valor: {metricas['analise_estatistica']['p_value']:.4f}")
                logger.info(f"Diferença estatisticamente significativa: "
                          f"{metricas['analise_estatistica']['significativo']}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")

def exemplo_avaliacao():
    """Exemplo de uso da avaliação online."""
    avaliador = AvaliacaoOnline()
    
    # Simular interações
    for i in range(100):
        # Alternar entre grupos A e B
        grupo = 'A' if i % 2 == 0 else 'B'
        
        # Simular recomendações e interações
        recomendacoes = [f'item_{j}' for j in range(5)]
        interacoes = [f'item_{j}' for j in range(np.random.randint(0, 4))]
        
        # Registrar interação
        avaliador.registrar_interacao(
            usuario_id=f'user_{i}',
            grupo=grupo,
            recomendacoes=recomendacoes,
            interacoes=interacoes,
            tempo_visualizacao=np.random.exponential(60),
            clicks=np.random.poisson(2)
        )
    
    # Gerar relatório
    avaliador.gerar_relatorio(janela_horas=1)

if __name__ == "__main__":
    configurar_logging()
    exemplo_avaliacao()