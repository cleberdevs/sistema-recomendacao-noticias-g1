import os
import sys
import time
import pandas as pd
import numpy as np
from memory_profiler import profile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class BenchmarkRecomendador:
    def __init__(self):
        self.preprocessador = None
        self.resultados = {
            'tempo_processamento': [],
            'uso_memoria': [],
            'tamanho_dados': []
        }

    @profile
    def testar_processamento(self, tamanho_chunk):
        """Testa o processamento com diferentes tamanhos de chunk."""
        try:
            self.preprocessador = PreProcessadorDadosSpark(
                memoria_executor="2g",
                memoria_driver="2g"
            )
            self.preprocessador.chunk_size = tamanho_chunk
            
            inicio = time.time()
            
            # Processar dados
            dados_treino, dados_itens = self.preprocessador.processar_dados_treino(
                ['dados/brutos/treino_parte_1.csv'],
                ['dados/brutos/itens/itens-parte1.csv']
            )
            
            tempo_total = time.time() - inicio
            
            # Registrar resultados
            self.resultados['tempo_processamento'].append(tempo_total)
            self.resultados['tamanho_dados'].append(len(dados_treino))
            
            return tempo_total
            
        finally:
            if self.preprocessador:
                self.preprocessador.__del__()

    def executar_benchmark(self):
        """Executa testes de benchmark completos."""
        tamanhos_chunk = [1000, 2000, 5000, 10000]
        
        logger.info("Iniciando benchmark")
        for tamanho in tamanhos_chunk:
            logger.info(f"\nTestando com tamanho de chunk: {tamanho}")
            tempo = self.testar_processamento(tamanho)
            logger.info(f"Tempo de processamento: {tempo:.2f} segundos")
        
        self._mostrar_resultados()

    def _mostrar_resultados(self):
        """Mostra resultados do benchmark."""
        df_resultados = pd.DataFrame(self.resultados)
        
        logger.info("\nResultados do Benchmark:")
        logger.info("\nTempos de Processamento:")
        logger.info(df_resultados[['tamanho_dados', 'tempo_processamento']])
        
        # Criar gráfico
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(df_resultados['tamanho_dados'], 
                    df_resultados['tempo_processamento'], 
                    marker='o')
            plt.xlabel('Tamanho dos Dados')
            plt.ylabel('Tempo de Processamento (s)')
            plt.title('Benchmark de Processamento')
            plt.grid(True)
            plt.savefig('benchmark_results.png')
            logger.info("Gráfico salvo em 'benchmark_results.png'")
        except Exception as e:
            logger.error(f"Erro ao criar gráfico: {str(e)}")

if __name__ == "__main__":
    configurar_logging()
    benchmark = BenchmarkRecomendador()
    benchmark.executar_benchmark()