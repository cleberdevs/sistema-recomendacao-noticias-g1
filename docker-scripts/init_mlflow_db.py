import mlflow
import os
import sys
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# Remove arquivo de banco de dados existente, se houver
if os.path.exists("mlflow.db"):
    os.remove("mlflow.db")

# Cria uma nova store e inicializa
try:
    store = SqlAlchemyStore("sqlite:///mlflow.db")
    # Força a inicialização das tabelas
    store.get_experiment_by_name("Default")
    print("Banco de dados MLflow inicializado com sucesso")
except Exception as e:
    print(f"Erro ao inicializar banco de dados MLflow: {e}")
    sys.exit(1)