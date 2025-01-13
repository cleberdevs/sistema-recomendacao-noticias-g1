import os
import sys
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple
import mlflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class ValidadorCruzado:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.preprocessador = None
        self.resultados = []

    def _preparar_dados(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara os dados para validação."""
        try:
            self.preprocessador = PreProcessadorDadosSpark(
                memoria_executor="2g",
                memoria_driver="2g"
            )
            
            return self.preprocessador.processar_dados_treino(
                ['dados/brutos/treino_parte_1.csv'],
                ['dados/brutos/itens/itens-parte1.csv']
            )
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            raise

    def _criar_folds_temporais(self, dados_treino: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Cria folds para validação cruzada temporal."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        folds = []
        
        # Ordenar por timestamp
        dados_treino['max_timestamp'] = dados_treino['historicoTimestamp'].apply(max)
        dados_treino = dados_treino.sort_values('max_timestamp')
        
        for train_idx, val_idx in tscv.split(dados_treino):
            train = dados_treino.iloc[train_idx]
            val = dados_treino.iloc[val_idx]
            folds.append((train, val))
        
        return folds

    def executar_validacao(self):
        """Executa validação cruzada."""
        try:
            # Preparar dados
            dados_treino, dados_itens = self._preparar_dados()
            
            # Criar folds
            folds = self._criar_folds_temporais(dados_treino)
            
            logger.info(f"Iniciando validação cruzada com {self.n_splits} folds")
            
            for i, (train, val) in enumerate(folds, 1):
                with mlflow.start_run(nested=True):
                    try:
                        logger.info(f"\nProcessando fold {i}/{self.n_splits}")
                        
                        # Treinar modelo
                        modelo = RecomendadorHibrido()
                        historia = modelo.treinar(train, dados_itens)
                        
                        # Avaliar no conjunto de validação
                        metricas_val = self._avaliar_modelo(modelo, val, dados_itens)
                        
                        # Registrar métricas
                        mlflow.log_metrics({
                            f"fold_{i}_accuracy": metricas_val['accuracy'],
                            f"fold_{i}_recall": metricas_val['recall']
                        })
                        
                        self.resultados.append(metricas_val)
                        
                    except Exception as e:
                        logger.error(f"Erro no fold {i}: {str(e)}")
                        continue
            
            self._mostrar_resultados()
            
        finally:
            if self.preprocessador:
                self.preprocessador.__del__()

    def _avaliar_modelo(self, modelo, dados_val, dados_itens):
        """Avalia o modelo no conjunto de validação."""
        metricas = {
            'accuracy': [],
            'recall': []
        }
        
        for _, usuario in dados_val.iterrows():
            historico_real = set(usuario['historico'])
            
            if len(historico_real) > 0:
                recomendacoes = set(modelo.prever(
                    usuario['idUsuario'],
                    n_recomendacoes=len(historico_real)
                ))
                
                # Calcular métricas
                intersecao = len(historico_real & recomendacoes)
                accuracy = intersecao / len(recomendacoes) if recomendacoes else 0
                recall = intersecao / len(historico_real) if historico_real else 0
                
                metricas['accuracy'].append(accuracy)
                metricas['recall'].append(recall)
        
        return {
            'accuracy': np.mean(metricas['accuracy']),
            'recall': np.mean(metricas['recall'])
        }

    def _mostrar_resultados(self):
        """Mostra resultados da validação cruzada."""
        logger.info("\nResultados da Validação Cruzada:")
        
        accuracy_media = np.mean([r['accuracy'] for r in self.resultados])
        recall_medio = np.mean([r['recall'] for r in self.resultados])
        
        logger.info(f"Accuracy média: {accuracy_media:.4f}")
        logger.info(f"Recall médio: {recall_medio:.4f}")
        
        # Salvar resultados detalhados
        import pandas as pd
        df_resultados = pd.DataFrame(self.resultados)
        df_resultados.to_csv('resultados_validacao.csv', index=False)
        logger.info("\nResultados detalhados salvos em 'resultados_validacao.csv'")

if __name__ == "__main__":
    configurar_logging()
    validador = ValidadorCruzado(n_splits=5)
    validador.executar_validacao()