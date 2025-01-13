import os
import sys
from sklearn.model_selection import ParameterGrid
import mlflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class OtimizadorParametros:
    def __init__(self):
        self.preprocessador = None
        self.melhor_score = float('-inf')
        self.melhores_params = None
        self.resultados = []

    def _preparar_dados(self):
        """Prepara os dados para otimização."""
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

    def otimizar(self):
        """Executa otimização de parâmetros."""
        try:
            # Preparar dados
            dados_treino, dados_itens = self._preparar_dados()
            
            # Definir grid de parâmetros
            param_grid = {
                'dim_embedding': [32, 64, 128],
                'learning_rate': [0.001, 0.01],
                'batch_size': [64, 128, 256],
                'dropout_rate': [0.2, 0.3, 0.4]
            }
            
            # Criar todas as combinações de parâmetros
            todas_combinacoes = list(ParameterGrid(param_grid))
            
            logger.info(f"Testando {len(todas_combinacoes)} combinações de parâmetros")
            
            # Testar cada combinação
            for params in todas_combinacoes:
                with mlflow.start_run(nested=True):
                    try:
                        logger.info(f"\nTestando parâmetros: {params}")
                        
                        # Registrar parâmetros
                        mlflow.log_params(params)
                        
                        # Treinar modelo
                        modelo = RecomendadorHibrido(**params)
                        historia = modelo.treinar(dados_treino, dados_itens)
                        
                        # Calcular métricas
                        metricas = {
                            'val_accuracy': historia.history['val_accuracy'][-1],
                            'val_loss': historia.history['val_loss'][-1]
                        }
                        
                        # Registrar métricas
                        mlflow.log_metrics(metricas)
                        
                        # Atualizar melhor resultado
                        if metricas['val_accuracy'] > self.melhor_score:
                            self.melhor_score = metricas['val_accuracy']
                            self.melhores_params = params
                            
                            # Salvar melhor modelo
                            modelo.salvar_modelo('modelos/modelos_salvos/melhor_modelo')
                        
                        # Registrar resultado
                        self.resultados.append({
                            'params': params,
                            'metricas': metricas
                        })
                        
                    except Exception as e:
                        logger.error(f"Erro ao testar parâmetros {params}: {str(e)}")
                        continue
            
            self._mostrar_resultados()
            
        finally:
            if self.preprocessador:
                self.preprocessador.__del__()

    def _mostrar_resultados(self):
        """Mostra resultados da otimização."""
        logger.info("\nResultados da Otimização:")
        logger.info(f"\nMelhores parâmetros encontrados: {self.melhores_params}")
        logger.info(f"Melhor score: {self.melhor_score:.4f}")
        
        # Criar DataFrame com resultados
        import pandas as pd
        resultados_df = pd.DataFrame([
            {**r['params'], **r['metricas']} 
            for r in self.resultados
        ])
        
        logger.info("\nTodos os resultados:")
        logger.info(resultados_df.sort_values('val_accuracy', ascending=False))
        
        # Salvar resultados
        resultados_df.to_csv('resultados_otimizacao.csv', index=False)
        logger.info("\nResultados salvos em 'resultados_otimizacao.csv'")

if __name__ == "__main__":
    configurar_logging()
    otimizador = OtimizadorParametros()
    otimizador.otimizar()