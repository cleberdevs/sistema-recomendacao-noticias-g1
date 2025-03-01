import mlflow
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MLflowConfig:
    def __init__(self, user_name: str = None):
        """
        Inicializa a configuração do MLflow.
        """
        self.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'recomendador_noticias')
        self.run_id = None
        self.ambiente = os.getenv('ENVIRONMENT', 'desenvolvimento')
        self.user_name = user_name or os.getenv('USER_NAME', 'sistema-recomendacao')
        self.active_run = None
        logger.info(f"MLflow configurado com URI: {self.tracking_uri}")

    def setup_mlflow(self):
        """Configura o MLflow para tracking de experimentos."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            if mlflow.active_run():
                logger.warning("Finalizando run MLflow ativo anterior")
                mlflow.end_run()
            
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
            
            logger.info("MLflow configurado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao configurar MLflow: {str(e)}")
            raise

    def log_tensorflow_model(self, modelo, nome_modelo: str, registered_model_name: str = None):
        """
        Registra um modelo TensorFlow no MLflow.
        """
        if not mlflow.active_run():
            logger.warning("Tentativa de registrar modelo sem run ativo")
            return
            
        try:
            logger.info(f"Registrando modelo TensorFlow: {nome_modelo}")
            mlflow.tensorflow.log_model(
                modelo,
                nome_modelo,
                registered_model_name=registered_model_name
            )
        except Exception as e:
            logger.error(f"Erro ao registrar modelo TensorFlow: {str(e)}")

    @contextmanager
    def iniciar_run(
        self, 
        run_name: Optional[str] = None, 
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """
        Gerencia o ciclo de vida de uma execução MLflow.
        
        Args:
            run_name: Nome opcional para a run
            tags: Dicionário com tags adicionais
            nested: Se True, permite runs aninhados
            
        Returns:
            MLflow ActiveRun
        """
        if run_name is None:
            run_name = f"execucao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        try:
            # Verificar se já existe um run ativo
            current_run = mlflow.active_run()
            if current_run and not nested:
                logger.warning(f"Finalizando run ativo anterior: {current_run.info.run_id}")
                mlflow.end_run()
            
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
            
            # Iniciar nova run
            self.active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=default_tags
            )
            self.run_id = self.active_run.info.run_id
            
            logger.info(f"Run iniciada com ID: {self.run_id}")
            
            try:
                yield self.active_run
            finally:
                if self.active_run:
                    self.finalizar_run()
                    
        except Exception as e:
            logger.error(f"Erro ao gerenciar MLflow run: {str(e)}")
            if self.active_run:
                self.finalizar_run(status="FAILED")
            raise

    def log_parametros(self, params: Dict[str, Any]):
        """
        Registra parâmetros no MLflow.
        
        Args:
            params: Dicionário com parâmetros para registrar
        """
        if not mlflow.active_run():
            logger.warning("Tentativa de registrar parâmetros sem run ativo")
            return
            
        try:
            logger.info("Registrando parâmetros")
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Erro ao registrar parâmetros: {str(e)}")

    def log_metricas(self, metrics: Dict[str, float]):
        """
        Registra métricas no MLflow.
        
        Args:
            metrics: Dicionário com métricas para registrar
        """
        if not mlflow.active_run():
            logger.warning("Tentativa de registrar métricas sem run ativo")
            return
            
        try:
            logger.info("Registrando métricas")
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Erro ao registrar métricas: {str(e)}")

    def log_modelo(self, modelo: Any, nome_modelo: str):
        """
        Salva o modelo no MLflow.
        
        Args:
            modelo: Modelo para salvar
            nome_modelo: Nome do modelo
        """
        if not mlflow.active_run():
            logger.warning("Tentativa de salvar modelo sem run ativo")
            return
            
        try:
            logger.info(f"Salvando modelo: {nome_modelo}")
            mlflow.sklearn.log_model(modelo, nome_modelo)
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")

    def log_artefato(self, caminho_arquivo: str):
        """
        Registra um artefato no MLflow.
        
        Args:
            caminho_arquivo: Caminho do arquivo para registrar
        """
        if not mlflow.active_run():
            logger.warning("Tentativa de registrar artefato sem run ativo")
            return
            
        try:
            logger.info(f"Registrando artefato: {caminho_arquivo}")
            mlflow.log_artifact(caminho_arquivo)
        except Exception as e:
            logger.error(f"Erro ao registrar artefato: {str(e)}")

    def finalizar_run(self, status: str = "FINISHED"):
        """
        Finaliza a run atual do MLflow.
        
        Args:
            status: Status final da run ("FINISHED" ou "FAILED")
        """
        try:
            if mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Finalizando run {run_id} com status: {status}")
                mlflow.end_run(status=status)
                self.active_run = None
                self.run_id = None
        except Exception as e:
            logger.error(f"Erro ao finalizar run: {str(e)}")

    def get_active_run_id(self) -> Optional[str]:
        """
        Retorna o ID da run ativa atual.
        
        Returns:
            str: ID da run ativa ou None se não houver run ativa
        """
        try:
            active_run = mlflow.active_run()
            return active_run.info.run_id if active_run else None
        except Exception as e:
            logger.error(f"Erro ao obter run ID ativo: {str(e)}")
            return None