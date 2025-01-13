import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.logging_config import configurar_logging
import logging

logger = logging.getLogger(__name__)

class MonitoramentoRecomendador:
    def __init__(self, port=8001):
        # Métricas Prometheus
        self.memoria_uso = Gauge('memoria_uso_bytes', 'Uso de memória em bytes')
        self.cpu_uso = Gauge('cpu_uso_percent', 'Uso de CPU em porcentagem')
        self.recomendacoes_total = Counter('recomendacoes_total', 'Total de recomendações geradas')
        self.tempo_processamento = Histogram(
            'tempo_processamento_segundos',
            'Tempo de processamento em segundos',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        # Iniciar servidor de métricas
        start_http_server(port)
        logger.info(f"Servidor de métricas iniciado na porta {port}")
        
        # Arquivo de log para métricas
        self.log_file = 'logs/metricas_producao.jsonl'

    def monitorar_recursos(self):
        """Monitora recursos do sistema."""
        try:
            processo = psutil.Process()
            
            # Memória
            uso_memoria = processo.memory_info().rss
            self.memoria_uso.set(uso_memoria)
            
            # CPU
            uso_cpu = processo.cpu_percent(interval=1.0)
            self.cpu_uso.set(uso_cpu)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memoria_bytes': uso_memoria,
                'cpu_percent': uso_cpu
            }
            
        except Exception as e:
            logger.error(f"Erro ao monitorar recursos: {str(e)}")
            return None

    def registrar_predicao(self, usuario_id: str, recomendacoes: list, tempo: float):
        """Registra métricas de uma predição."""
        try:
            # Incrementar contador
            self.recomendacoes_total.inc()
            
            # Registrar tempo
            self.tempo_processamento.observe(tempo)
            
            # Registrar em arquivo
            dados = {
                'timestamp': datetime.now().isoformat(),
                'usuario_id': usuario_id,
                'num_recomendacoes': len(recomendacoes),
                'tempo_processamento': tempo
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(dados) + '\n')
                
        except Exception as e:
            logger.error(f"Erro ao registrar predição: {str(e)}")

    def gerar_relatorio_diario(self):
        """Gera relatório diário de métricas."""
        try:
            # Ler logs do dia
            hoje = datetime.now().date()
            metricas = []
            
            with open(self.log_file, 'r') as f:
                for linha in f:
                    dados = json.loads(linha)
                    data = datetime.fromisoformat(dados['timestamp']).date()
                    if data == hoje:
                        metricas.append(dados)
            
            if not metricas:
                logger.warning("Sem dados para relatório")
                return
            
            df = pd.DataFrame(metricas)
            
            # Calcular estatísticas
            estatisticas = {
                'total_predicoes': len(df),
                'tempo_medio': df['tempo_processamento'].mean(),
                'tempo_max': df['tempo_processamento'].max(),
                'tempo_min': df['tempo_processamento'].min(),
                'recomendacoes_media': df['num_recomendacoes'].mean()
            }
            
            # Salvar relatório
            nome_arquivo = f'logs/relatorio_{hoje.strftime("%Y%m%d")}.json'
            with open(nome_arquivo, 'w') as f:
                json.dump(estatisticas, f, indent=4)
                
            logger.info(f"Relatório gerado: {nome_arquivo}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")

    def monitoramento_continuo(self, intervalo=60):
        """Executa monitoramento contínuo."""
        logger.info("Iniciando monitoramento contínuo")
        ultima_geracao_relatorio = datetime.now().date()
        
        try:
            while True:
                # Monitorar recursos
                metricas = self.monitorar_recursos()
                if metricas:
                    with open('logs/recursos.jsonl', 'a') as f:
                        f.write(json.dumps(metricas) + '\n')
                
                # Verificar se precisa gerar relatório
                hoje = datetime.now().date()
                if hoje > ultima_geracao_relatorio:
                    self.gerar_relatorio_diario()
                    ultima_geracao_relatorio = hoje
                
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            logger.info("Monitoramento interrompido")
        except Exception as e:
            logger.error(f"Erro no monitoramento: {str(e)}")
            raise

def exemplo_uso():
    """Exemplo de uso do monitoramento."""
    monitoramento = MonitoramentoRecomendador()
    
    # Simular algumas predições
    for i in range(5):
        tempo_inicio = time.time()
        time.sleep(np.random.random())  # Simular processamento
        
        monitoramento.registrar_predicao(
            usuario_id=f"user_{i}",
            recomendacoes=[f"item_{j}" for j in range(5)],
            tempo=time.time() - tempo_inicio
        )
    
    # Gerar relatório
    monitoramento.gerar_relatorio_diario()
    
    # Iniciar monitoramento contínuo
    monitoramento.monitoramento_continuo(intervalo=10)

if __name__ == "__main__":
    configurar_logging()
    exemplo_uso()