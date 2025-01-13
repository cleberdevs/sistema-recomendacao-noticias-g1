from flask import Flask, request, jsonify
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os

# Configurar logging
configurar_logging()
logger = get_logger(__name__)

app = Flask(__name__)

# Carregar modelo e dados
try:
    logger.info("Inicializando API e carregando modelo...")
    modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    dados_itens = pd.read_csv('dados/brutos/itens/itens-parte1.csv')
    logger.info("Modelo e dados carregados com sucesso")
except Exception as e:
    logger.error(f"Erro fatal ao inicializar API: {str(e)}")
    raise

@app.before_request
def antes_da_requisicao():
    """Log de todas as requisições."""
    logger.info(f"Requisição recebida: {request.method} {request.path}")

@app.after_request
def depois_da_requisicao(response):
    """Log das respostas."""
    logger.info(f"Resposta enviada: Status {response.status_code}")
    return response

@app.route('/saude', methods=['GET'])
def verificar_saude():
    """Endpoint para verificação de saúde da API."""
    try:
        if modelo is None:
            return jsonify({
                "status": "erro",
                "mensagem": "Modelo não carregado"
            }), 500
            
        return jsonify({
            "status": "saudavel",
            "versao_modelo": os.getenv('MODEL_VERSION', 'v1'),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Erro na verificação de saúde: {str(e)}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

@app.route('/prever', methods=['POST'])
@tratar_excecoes
@validar_entrada_json(['id_usuario'])
def obter_recomendacoes():
    """Endpoint para obter recomendações."""
    try:
        dados = request.get_json()
        id_usuario = dados['id_usuario']
        n_recomendacoes = dados.get('n_recomendacoes', 10)
        
        logger.info(f"Gerando recomendações para usuário: {id_usuario}")
        
        # Registrar predição no MLflow
        with mlflow.start_run(run_name=f"predicao_{id_usuario}"):
            recomendacoes = modelo.prever(id_usuario, n_recomendacoes)
            
            mlflow.log_params({
                "id_usuario": id_usuario,
                "n_recomendacoes": n_recomendacoes
            })
            
            mlflow.log_metrics({
                "num_recomendacoes": len(recomendacoes)
            })
        
        return jsonify({
            "recomendacoes": recomendacoes,
            "metadata": {
                "usuario": id_usuario,
                "quantidade": len(recomendacoes),
                "timestamp": datetime.now().isoformat(),
                "versao_modelo": os.getenv('MODEL_VERSION', 'v1')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.errorhandler(404)
def nao_encontrado(erro):
    """Handler para rotas não encontradas."""
    logger.warning(f"Rota não encontrada: {request.path}")
    return jsonify({"erro": "Rota não encontrada"}), 404

@app.errorhandler(500)
def erro_interno(erro):
    """Handler para erros internos."""
    logger.error(f"Erro interno do servidor: {str(erro)}")
    return jsonify({
        "erro": "Erro interno do servidor",
        "detalhes": str(erro)
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)