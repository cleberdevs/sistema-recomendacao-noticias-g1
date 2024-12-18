from mlflow.server import app
import os

if __name__ == "__main__":
    os.makedirs("mlflow-artifacts", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
