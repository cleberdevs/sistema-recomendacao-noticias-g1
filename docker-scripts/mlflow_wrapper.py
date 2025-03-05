#!/usr/bin/env python3
import os
import sys
import importlib

# Configurações do MLflow que queremos injetar
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

def enhance_mlflow():
    """Melhora o módulo MLflow sem substituí-lo completamente"""
    import mlflow
    
    # Armazena a função original para usar em nosso wrapper
    original_end_run = mlflow.end_run
    
    # Define uma versão segura de end_run
    def safe_end_run(*args, **kwargs):
        try:
            return original_end_run(*args, **kwargs)
        except Exception as e:
            print(f"Erro ao finalizar MLflow run: {e}")
            return None
    
    # Substitui a função original
    mlflow.end_run = safe_end_run
    
    # Armazena a função original log_artifacts
    original_log_artifacts = mlflow.log_artifacts
    
    # Define uma função para garantir o registro do modelo
    def ensure_model_registered(artifact_path, run_id=None):
        try:
            if run_id is None and mlflow.active_run() is not None:
                run_id = mlflow.active_run().info.run_id
            if run_id is not None:
                mlflow.register_model(f"runs:/{run_id}/{artifact_path}", "sistema_recomendacao")
                print("Modelo registrado com sucesso no MLflow")
        except Exception as e:
            print(f"Erro ao registrar modelo no MLflow: {e}")
    
    # Define uma nova versão de log_artifacts que também registra o modelo
    def log_artifacts_with_registry(local_dir, artifact_path=None):
        result = original_log_artifacts(local_dir, artifact_path)
        ensure_model_registered("model")
        return result
    
    # Substitui a função original
    mlflow.log_artifacts = log_artifacts_with_registry
    
    print("MLflow melhorado com funções adicionais de segurança e registro automático de modelos")

def setup_project_modules():
    """Configura módulos específicos do projeto, se existirem"""
    try:
        import src.config.mlflow_config
        print("Módulo src.config.mlflow_config encontrado e configurado")
    except ImportError:
        print("Módulo src.config.mlflow_config não encontrado, pulando...")
    
    try:
        import src.modelo.recomendador
        print("Módulo src.modelo.recomendador encontrado e configurado")
    except ImportError:
        print("Módulo src.modelo.recomendador não encontrado, pulando...")

def run_with_mlflow_enhancement(script_path):
    """Executa o script com as melhorias do MLflow"""
    # Adiciona o diretório do script ao path
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Adiciona o diretório atual ao path, se ainda não estiver
    current_dir = os.path.abspath(os.getcwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Garante que a variável de ambiente esteja definida
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    print(f"MLFLOW_TRACKING_URI configurado como: {os.environ['MLFLOW_TRACKING_URI']}")
    
    # Configura o MLflow e outros módulos
    enhance_mlflow()
    setup_project_modules()
    
    # Executa o script como um módulo em vez de usar exec
    print(f"Executando {script_path} como um módulo...")
    
    # Salva os argumentos originais
    original_argv = sys.argv.copy()
    
    try:
        # Configura sys.argv para o script
        sys.argv = [script_path] + sys.argv[1:]
        
        # Carrega o script como um módulo
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
    finally:
        # Restaura os argumentos originais
        sys.argv = original_argv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python mlflow_wrapper.py <script.py> [args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    # Trata erros para melhor diagnóstico
    try:
        run_with_mlflow_enhancement(script_path)
    except Exception as e:
        import traceback
        print(f"Erro durante a execução do wrapper MLflow: {e}")
        traceback.print_exc()
        sys.exit(1)