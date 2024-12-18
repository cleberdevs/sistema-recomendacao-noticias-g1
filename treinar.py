import os
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento import PreProcessadorDados
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig

def treinar_modelo():
    # Configurar MLflow
    mlflow_config = MLflowConfig()
    mlflow_config.setup_mlflow()
    
    with mlflow_config.iniciar_run(run_name="treinamento_completo"):
        # Configurar diretórios
        criar_diretorio_se_nao_existe('modelos/modelos_salvos')
        
        # Carregar dados
        arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte_*.csv')
        arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
        
        # Preprocessar dados
        preprocessador = PreProcessadorDados()
        dados_treino, dados_itens = preprocessador.processar_dados_treino(
            arquivos_treino,
            arquivos_itens
        )
        dados_itens = preprocessador.preparar_features_texto(dados_itens)
        
        # Registrar métricas de preprocessamento
        metricas_prep = preprocessador.calcular_metricas_preprocessamento(
            dados_treino,
            dados_itens
        )
        mlflow_config.log_parametros(metricas_prep)
        
        # Criar e treinar modelo
        modelo = RecomendadorHibrido()
        historia_treino = modelo.treinar(dados_treino, dados_itens)
        
        # Salvar modelo
        caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
        modelo.salvar_modelo(caminho_modelo)
        mlflow_config.log_artefato(caminho_modelo)

if __name__ == "__main__":
    treinar_modelo()
