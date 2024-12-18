import mlflow
import os
from datetime import datetime

class MLflowConfig:
    def __init__(self):
        self.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = "recomendador_noticias"
        
    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            experimento = mlflow.get_experiment_by_name(self.experiment_name)
            if experimento is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Erro ao configurar MLflow: {str(e)}")
        
    def iniciar_run(self, run_name=None):
        if run_name is None:
            run_name = f"execucao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
        
    def log_parametros(self, params):
        mlflow.log_params(params)
        
    def log_metricas(self, metrics):
        mlflow.log_metrics(metrics)
        
    def log_modelo(self, modelo, nome_modelo):
        mlflow.sklearn.log_model(modelo, nome_modelo)
        
    def log_artefato(self, caminho_arquivo):
        mlflow.log_artifact(caminho_arquivo)
