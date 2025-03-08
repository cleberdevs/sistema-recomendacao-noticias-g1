import mlflow
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Adicionando configurações explícitas para evitar dependência do Spark
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
os.environ["MLFLOW_REGISTRY_URI"] = os.getenv('MLFLOW_REGISTRY_URI', 'sqlite:///mlflow.db')
# Desativa a resolução automática de URI do registro via Spark
os.environ["MLFLOW_ENABLE_LAZY_REGISTRY"] = "true"

class MLflowConfig:
    def __init__(self, user_name: str = None):
        """
        Inicializa a configuração do MLflow.
        """
        self.tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        self.registry_uri = os.environ["MLFLOW_REGISTRY_URI"]
        self.experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'recomendador_noticias')
        self.run_id = None
        self.ambiente = os.getenv('ENVIRONMENT', 'desenvolvimento')
        self.user_name = user_name or os.getenv('USER_NAME', 'sistema-recomendacao')
        self.active_run = None
        logger.info(f"MLflow configurado com URI: {self.tracking_uri}, Registry URI: {self.registry_uri}")

    def setup_mlflow(self):
        """Configura o MLflow para tracking de experimentos com tratamento robusto de erros."""
        try:
            # Configurar URIs explicitamente
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)
            
            # Finalizar run ativo se existir
            try:
                if mlflow.active_run():
                    logger.warning("Finalizando run MLflow ativo anterior")
                    mlflow.end_run()
            except Exception as e:
                logger.warning(f"Erro ao finalizar run anterior: {str(e)}")
            
            # Verificar ou criar experimento
            try:
                experimento = mlflow.get_experiment_by_name(self.experiment_name)
                if experimento is None:
                    logger.info(f"Criando novo experimento: {self.experiment_name}")
                    mlflow.create_experiment(
                        self.experiment_name,
                        tags={
                            "ambiente": self.ambiente,
                            "created_by": self.user_name
                        }
                    )
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                logger.error(f"Erro ao configurar experimento: {str(e)}")
                # Continua mesmo com erro
            
            logger.info("MLflow configurado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao configurar MLflow: {str(e)}")
            # Não propaga o erro - permite que a aplicação continue sem MLflow

    def log_tensorflow_model(self, modelo, nome_modelo: str, registered_model_name: str = None):
        """
        Registra um modelo TensorFlow no MLflow com tratamento robusto de erros.
        """
        if not self._check_mlflow_active():
            return
            
        try:
            logger.info(f"Registrando modelo TensorFlow: {nome_modelo}")
            mlflow.tensorflow.log_model(
                modelo,
                nome_modelo,
                registered_model_name=registered_model_name
            )
            logger.info(f"Modelo TensorFlow registrado com sucesso: {nome_modelo}")
            return True
        except Exception as e:
            logger.error(f"Erro ao registrar modelo TensorFlow: {str(e)}")
            return False

    def _check_mlflow_active(self) -> bool:
        """Verifica se há um run MLflow ativo de forma segura"""
        try:
            active_run = mlflow.active_run()
            if not active_run:
                logger.warning("Nenhum run MLflow ativo")
                return False
            return True
        except Exception as e:
            logger.warning(f"Erro ao verificar run MLflow ativo: {str(e)}")
            return False

    @contextmanager
    def iniciar_run(
        self, 
        run_name: Optional[str] = None, 
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """
        Gerencia o ciclo de vida de uma execução MLflow com tratamento robusto de erros.
        
        Args:
            run_name: Nome opcional para a run
            tags: Dicionário com tags adicionais
            nested: Se True, permite runs aninhados
            
        Returns:
            MLflow ActiveRun ou None em caso de erro
        """
        if run_name is None:
            run_name = f"execucao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        try:
            # Verificar se já existe um run ativo
            try:
                current_run = mlflow.active_run()
                if current_run and not nested:
                    logger.warning(f"Finalizando run ativo anterior: {current_run.info.run_id}")
                    mlflow.end_run()
            except Exception as e:
                logger.warning(f"Erro ao verificar/finalizar run ativo: {str(e)}")
            
            logger.info(f"Iniciando MLflow run: {run_name}")
            
            # Tags padrão
            default_tags = {
                "ambiente": self.ambiente,
                "versao_modelo": os.getenv('MODEL_VERSION', 'v1'),
                "timestamp": datetime.now().isoformat(),
                "nested": str(nested),
                "mlflow.user": self.user_name,
                "created_by": self.user_name
            }
            
            # Combinar com tags adicionais
            if tags:
                default_tags.update(tags)
            
            # Iniciar nova run com tratamento de erro
            try:
                self.active_run = mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    tags=default_tags
                )
                self.run_id = self.active_run.info.run_id
                logger.info(f"Run iniciada com ID: {self.run_id}")
            except Exception as e:
                logger.error(f"Erro ao iniciar MLflow run: {str(e)}")
                self.active_run = None
                self.run_id = None
            
            try:
                yield self.active_run
            finally:
                if self.active_run:
                    self.finalizar_run()
                    
        except Exception as e:
            logger.error(f"Erro ao gerenciar MLflow run: {str(e)}")
            if self.active_run:
                self.finalizar_run(status="FAILED")
            # Não propaga a exceção - retorna None

    def log_parametros(self, params: Dict[str, Any]):
        """
        Registra parâmetros no MLflow com tratamento robusto de erros.
        
        Args:
            params: Dicionário com parâmetros para registrar
        """
        if not self._check_mlflow_active():
            return
            
        try:
            logger.info("Registrando parâmetros")
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Erro ao registrar parâmetros: {str(e)}")

    def log_metricas(self, metrics: Dict[str, float]):
        """
        Registra métricas no MLflow com tratamento robusto de erros.
        
        Args:
            metrics: Dicionário com métricas para registrar
        """
        if not self._check_mlflow_active():
            return
            
        try:
            logger.info("Registrando métricas")
            # Converte valores para float se necessário
            clean_metrics = {}
            for k, v in metrics.items():
                try:
                    clean_metrics[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Ignorando métrica não-numérica: {k}={v}")
            
            mlflow.log_metrics(clean_metrics)
        except Exception as e:
            logger.error(f"Erro ao registrar métricas: {str(e)}")

    def log_modelo(self, modelo: Any, nome_modelo: str):
        """
        Salva o modelo no MLflow com tratamento robusto de erros.
        
        Args:
            modelo: Modelo para salvar
            nome_modelo: Nome do modelo
        """
        if not self._check_mlflow_active():
            return
            
        try:
            logger.info(f"Salvando modelo: {nome_modelo}")
            mlflow.sklearn.log_model(modelo, nome_modelo)
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")

    def log_artefato(self, caminho_arquivo: str):
        """
        Registra um artefato no MLflow com tratamento robusto de erros.
        
        Args:
            caminho_arquivo: Caminho do arquivo para registrar
        """
        if not self._check_mlflow_active():
            return
            
        try:
            if not os.path.exists(caminho_arquivo):
                logger.warning(f"Artefato não encontrado: {caminho_arquivo}")
                return
                
            logger.info(f"Registrando artefato: {caminho_arquivo}")
            mlflow.log_artifact(caminho_arquivo)
        except Exception as e:
            logger.error(f"Erro ao registrar artefato: {str(e)}")

    def finalizar_run(self, status: str = "FINISHED"):
        """
        Finaliza a run atual do MLflow com tratamento robusto de erros.
        
        Args:
            status: Status final da run ("FINISHED" ou "FAILED")
        """
        try:
            run_id = None
            try:
                active_run = mlflow.active_run()
                if active_run:
                    run_id = active_run.info.run_id
            except Exception as e:
                logger.warning(f"Erro ao obter run ativo: {str(e)}")
            
            if run_id:
                logger.info(f"Finalizando run {run_id} com status: {status}")
                try:
                    mlflow.end_run(status=status)
                    self.active_run = None
                    self.run_id = None
                except Exception as e:
                    logger.error(f"Erro ao finalizar run: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao finalizar run: {str(e)}")

    def get_active_run_id(self) -> Optional[str]:
        """
        Retorna o ID da run ativa atual de forma segura.
        
        Returns:
            str: ID da run ativa ou None se não houver run ativa
        """
        try:
            active_run = mlflow.active_run()
            return active_run.info.run_id if active_run else None
        except Exception as e:
            logger.error(f"Erro ao obter run ID ativo: {str(e)}")
            return None