import os
import sys
import time
import signal
import subprocess
import requests

def start_mlflow_server():
    """Inicia o servidor MLflow e garante que continue rodando"""
    # Inicia servidor MLflow
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    # Abre arquivo de log
    log_file = open("mlflow.log", "w")
    
    # Inicia processo
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"MLflow iniciado com PID {proc.pid}")
    
    # Aguarda servidor estar pronto
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/")
            if response.status_code == 200:
                print(f"Servidor MLflow está rodando (tentativa {attempt+1}/{max_attempts})")
                return proc
        except requests.exceptions.ConnectionError:
            pass
        
        # Verifica se o processo ainda está rodando
        if proc.poll() is not None:
            print(f"Processo MLflow encerrou com código {proc.returncode}")
            log_file.close()
            with open("mlflow.log", "r") as f:
                print(f.read())
            sys.exit(1)
        
        time.sleep(1)
    
    print("Servidor MLflow falhou em iniciar no tempo esperado")
    proc.terminate()
    log_file.close()
    sys.exit(1)

def verify_model_registration():
    """Verifica se os modelos foram registrados no MLflow"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    print(f"Encontrados {len(experiments)} experimentos no MLflow")
    for exp in experiments:
        print(f"Experimento: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(exp.experiment_id)
        print(f"  Encontradas {len(runs)} execuções")
        
        for run in runs:
            print(f"  ID da Execução: {run.info.run_id}, Status: {run.info.status}")
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"  Artefatos: {len(artifacts)}")
            for artifact in artifacts:
                print(f"    {artifact.path}")
    
    return True

if __name__ == "__main__":
    # Verifica se está sendo chamado para verificar registro de modelo
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        success = verify_model_registration()
        sys.exit(0 if success else 1)
    
    # Inicia servidor MLflow
    mlflow_proc = start_mlflow_server()
    
    # Configura manipuladores de sinal para manter MLflow rodando
    def handle_signal(sig, frame):
        if sig in (signal.SIGINT, signal.SIGTERM):
            print("Sinal de terminação recebido, mas mantendo MLflow rodando para o pipeline")
        else:
            print(f"Sinal recebido {sig}")
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Mantém processo rodando até ser explicitamente encerrado
    try:
        while mlflow_proc.poll() is None:
            time.sleep(1)
        
        print(f"MLflow encerrou com código {mlflow_proc.returncode}")
        with open("mlflow.log", "r") as f:
            print(f.read())
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
        mlflow_proc.terminate()