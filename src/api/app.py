from flask import Flask, request, jsonify
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json
import mlflow

app = Flask(__name__)

# Carregar modelo do MLflow
# cliente_mlflow = mlflow.tracking.MlflowClient()
# modelo_producao = cliente_mlflow.get_latest_versions("recomendador_hibrido", stages=["Production"])[0]
modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
dados_itens = pd.read_csv('dados/brutos/itens/itens-parte1.csv')

@app.route('/saude', methods=['GET'])
def verificar_saude():
    return jsonify({"status": "saudavel"}), 200

@app.route('/prever', methods=['POST'])
@tratar_excecoes
@validar_entrada_json(['id_usuario'])
def obter_recomendacoes():
    dados = request.get_json()
    id_usuario = dados['id_usuario']
    n_recomendacoes = dados.get('n_recomendacoes', 10)
    
    recomendacoes = modelo.prever(id_usuario, n_recomendacoes)
    
    return jsonify({
        "recomendacoes": recomendacoes
    }), 200

@app.errorhandler(404)
def nao_encontrado(erro):
    return jsonify({"erro": "Rota não encontrada"}), 404

@app.errorhandler(500)
def erro_interno(erro):
    return jsonify({"erro": "Erro interno do servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
