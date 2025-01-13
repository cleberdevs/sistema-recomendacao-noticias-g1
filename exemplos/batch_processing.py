import os
import sys
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modelo.recomendador import RecomendadorHibrido
from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, 
                 tamanho_batch: int = 100, 
                 num_threads: int = 4,
                 timeout: int = 30):
        self.tamanho_batch = tamanho_batch
        self.num_threads = num_threads
        self.timeout = timeout
        self.fila_processar = queue.Queue()
        self.fila_resultados = queue.Queue()
        self.modelo = None
        self.stop_flag = threading.Event()

    def carregar_modelo(self, caminho_modelo: str):
        """Carrega o modelo de recomendação."""
        try:
            logger.info(f"Carregando modelo de {caminho_modelo}")
            self.modelo = RecomendadorHibrido.carregar_modelo(caminho_modelo)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise

    def processar_lote(self, usuarios: List[Dict]) -> pd.DataFrame:
        """
        Processa um lote de usuários em paralelo.
        
        Args:
            usuarios: Lista de dicionários com informações dos usuários
            
        Returns:
            DataFrame com recomendações
        """
        try:
            # Limpar filas
            while not self.fila_processar.empty():
                self.fila_processar.get()
            while not self.fila_resultados.empty():
                self.fila_resultados.get()
            
            # Resetar flag de parada
            self.stop_flag.clear()
            
            # Adicionar usuários à fila
            for usuario in usuarios:
                self.fila_processar.put(usuario)
            
            # Iniciar workers
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(self._worker)
                    for _ in range(self.num_threads)
                ]
                
                # Aguardar conclusão ou timeout
                resultados = []
                try:
                    for future in as_completed(futures, timeout=self.timeout):
                        resultados.extend(future.result())
                except TimeoutError:
                    logger.warning("Timeout no processamento do lote")
                    self.stop_flag.set()
                
            return pd.DataFrame(resultados)
            
        except Exception as e:
            logger.error(f"Erro no processamento em lote: {str(e)}")
            raise

    def _worker(self) -> List[Dict]:
        """Worker thread para processamento."""
        resultados = []
        
        while not self.stop_flag.is_set():
            try:
                # Tentar obter próximo usuário
                usuario = self.fila_processar.get_nowait()
            except queue.Empty:
                break
                
            try:
                # Gerar recomendações
                recomendacoes = self.modelo.prever(
                    usuario['id_usuario'],
                    n_recomendacoes=usuario.get('n_recomendacoes', 10)
                )
                
                # Registrar resultado
                resultados.append({
                    'id_usuario': usuario['id_usuario'],
                    'recomendacoes': recomendacoes,
                    'status': 'sucesso',
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Erro ao processar usuário {usuario['id_usuario']}: {str(e)}")
                resultados.append({
                    'id_usuario': usuario['id_usuario'],
                    'recomendacoes': [],
                    'status': 'erro',
                    'erro': str(e),
                    'timestamp': time.time()
                })
                
            finally:
                self.fila_processar.task_done()
        
        return resultados

def exemplo_batch():
    """Exemplo de uso do processamento em batch."""
    # Criar processador
    processor = BatchProcessor(
        tamanho_batch=100,
        num_threads=4
    )
    
    # Carregar modelo
    processor.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    
    # Criar dados de exemplo
    usuarios = [
        {'id_usuario': f'user_{i}', 'n_recomendacoes': 5}
        for i in range(1000)
    ]
    
    # Processar em lotes
    resultados_todos = []
    for i in range(0, len(usuarios), processor.tamanho_batch):
        lote = usuarios[i:i + processor.tamanho_batch]
        logger.info(f"Processando lote {i//processor.tamanho_batch + 1}")
        
        resultados = processor.processar_lote(lote)
        resultados_todos.append(resultados)
    
    # Combinar resultados
    df_final = pd.concat(resultados_todos, ignore_index=True)
    
    # Mostrar estatísticas
    logger.info("\nEstatísticas do processamento:")
    logger.info(f"Total de usuários processados: {len(df_final)}")
    logger.info(f"Sucessos: {len(df_final[df_final['status'] == 'sucesso'])}")
    logger.info(f"Erros: {len(df_final[df_final['status'] == 'erro'])}")
    
    # Salvar resultados
    df_final.to_csv('resultados_batch.csv', index=False)
    logger.info("\nResultados salvos em 'resultados_batch.csv'")

if __name__ == "__main__":
    configurar_logging()
    exemplo_batch()