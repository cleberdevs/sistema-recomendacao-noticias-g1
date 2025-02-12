'''from flask import Flask, request, jsonify
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging

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
    app.run(host='0.0.0.0', port=port)'''

'''from flask import Flask, request, jsonify, render_template
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os
from pyspark.sql import SparkSession
import gc

# Configurar logging
configurar_logging()
logger = get_logger(__name__)

app = Flask(__name__)

# Carregar modelo e dados
try:
    logger.info("Inicializando API e carregando modelo...")
    
    # Limpar memória
    gc.collect()
    
    # Carregar modelo
    modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    dados_itens = pd.read_csv('dados/brutos/itens/itens-parte1.csv')
    
    # Inicializar Spark
    spark = SparkSession.builder \
        .appName("RecomendadorAPI") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    logger.info("Modelo, dados e Spark carregados com sucesso")
except Exception as e:
    logger.error(f"Erro fatal ao inicializar API: {str(e)}")
    raise

def carregar_dados_usuarios():
    """
    Carrega e retorna lista de usuários disponíveis.
    """
    try:
        usuarios_modelo = sorted(modelo.usuario_id_to_index.keys())
        usuarios_info = []
        
        for usuario_id in usuarios_modelo:
            n_historico = len(modelo.itens_usuario.get(usuario_id, []))
            usuarios_info.append({
                'id': usuario_id,
                'n_historico': n_historico
            })
        
        return usuarios_info
    except Exception as e:
        logger.error(f"Erro ao carregar dados dos usuários: {str(e)}")
        return []

@app.before_request
def antes_da_requisicao():
    """Log de todas as requisições."""
    logger.info(f"Requisição recebida: {request.method} {request.path}")

@app.after_request
def depois_da_requisicao(response):
    """Log das respostas."""
    logger.info(f"Resposta enviada: Status {response.status_code}")
    return response

@app.route('/')
def pagina_inicial():
    """Página inicial com interface para busca de recomendações."""
    try:
        # Parâmetros de paginação
        page = int(request.args.get('page', 1))
        per_page = 20  # usuários por página
        
        # Carregar todos os usuários
        todos_usuarios = carregar_dados_usuarios()
        total_usuarios = len(todos_usuarios)
        
        # Calcular total de páginas
        total_pages = (total_usuarios + per_page - 1) // per_page
        
        # Garantir que a página está dentro dos limites
        page = max(1, min(page, total_pages))
        
        # Selecionar usuários da página atual
        inicio = (page - 1) * per_page
        fim = inicio + per_page
        usuarios_pagina = todos_usuarios[inicio:fim]
        
        return render_template('index.html', 
                             usuarios=usuarios_pagina,
                             current_page=page,
                             total_pages=total_pages)
                             
    except Exception as e:
        logger.error(f"Erro ao renderizar página inicial: {str(e)}")
        return render_template('index.html', 
                             usuarios=[],
                             current_page=1,
                             total_pages=1,
                             erro=str(e))

@app.route('/buscar_usuario', methods=['POST'])
def buscar_usuario():
    """Endpoint para buscar detalhes de um usuário específico."""
    try:
        usuario_id = request.form.get('usuario_id')
        if not usuario_id:
            return jsonify({"erro": "ID do usuário não fornecido"}), 400
            
        if usuario_id not in modelo.usuario_id_to_index:
            return jsonify({"erro": "Usuário não encontrado"}), 404
            
        # Obter histórico
        historico = modelo.itens_usuario.get(usuario_id, [])
        urls_historico = []
        for idx in list(historico)[-5:]:  # Últimos 5 itens
            if idx in modelo.index_to_item_id:
                urls_historico.append(modelo.index_to_item_id[idx])
                
        # Gerar recomendações
        recomendacoes = modelo.prever(usuario_id, k=5)
        
        # Log no MLflow
        with mlflow.start_run(run_name=f"web_predicao_{usuario_id}"):
            mlflow.log_params({
                "usuario_id": usuario_id,
                "n_historico": len(historico)
            })
            mlflow.log_metrics({
                "n_recomendacoes": len(recomendacoes)
            })
        
        return jsonify({
            "usuario_id": usuario_id,
            "n_historico": len(historico),
            "ultimos_itens": urls_historico,
            "recomendacoes": recomendacoes
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao buscar usuário: {str(e)}")
        return jsonify({"erro": str(e)}), 500

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
    """Endpoint para obter recomendações via API."""
    try:
        dados = request.get_json()
        id_usuario = dados['id_usuario']
        n_recomendacoes = dados.get('n_recomendacoes', 10)
        
        logger.info(f"Gerando recomendações para usuário: {id_usuario}")
        
        # Registrar predição no MLflow
        with mlflow.start_run(run_name=f"api_predicao_{id_usuario}"):
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

@app.route('/usuarios', methods=['GET'])
def listar_usuarios():
    """Endpoint para listar usuários disponíveis."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        usuarios = carregar_dados_usuarios()
        total = len(usuarios)
        
        inicio = (page - 1) * per_page
        fim = inicio + per_page
        
        return jsonify({
            "usuarios": usuarios[inicio:fim],
            "metadata": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }), 200
    except Exception as e:
        logger.error(f"Erro ao listar usuários: {str(e)}")
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

@app.teardown_appcontext
def limpar_recursos(error):
    """Limpa recursos ao encerrar a aplicação."""
    try:
        if 'spark' in globals():
            spark.stop()
        gc.collect()
    except Exception as e:
        logger.error(f"Erro ao limpar recursos: {str(e)}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)'''

from flask import Flask, request, jsonify, render_template
from src.modelo.recomendador import RecomendadorHibrido
import pandas as pd
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os
from pyspark.sql import SparkSession
import gc
from prever import fazer_previsoes, carregar_modelo, mostrar_detalhes_usuario

# Configurar logging
configurar_logging()
logger = get_logger(__name__)

# Verificar estrutura de diretórios
app_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(app_dir, 'templates')

if not os.path.exists(templates_dir):
    raise RuntimeError(
        f"Diretório de templates não encontrado em {templates_dir}. "
        "Crie a pasta 'templates' no mesmo nível do app.py"
    )

if not os.path.exists(os.path.join(templates_dir, 'index.html')):
    raise RuntimeError(
        f"Arquivo index.html não encontrado em {templates_dir}. "
        "Certifique-se de criar o arquivo templates/index.html"
    )

app = Flask(__name__)

# Carregar modelo e dados
try:
    logger.info("Inicializando API e carregando modelo...")
    
    # Limpar memória
    gc.collect()
    
    # Carregar modelo
    modelo = carregar_modelo()
    
    # Inicializar Spark
    spark = SparkSession.builder \
        .appName("RecomendadorAPI") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    logger.info("Modelo e Spark carregados com sucesso")
except Exception as e:
    logger.error(f"Erro fatal ao inicializar API: {str(e)}")
    raise

def carregar_dados_usuarios():
    """
    Carrega e retorna lista de usuários disponíveis com informações sobre histórico.
    """
    try:
        usuarios_modelo = sorted(modelo.usuario_id_to_index.keys())
        usuarios_info = []
        
        for usuario_id in usuarios_modelo:
            n_historico = len(modelo.itens_usuario.get(usuario_id, []))
            usuarios_info.append({
                'id': usuario_id,
                'n_historico': n_historico
            })
        
        return usuarios_info
    except Exception as e:
        logger.error(f"Erro ao carregar dados dos usuários: {str(e)}")
        return []

@app.before_request
def antes_da_requisicao():
    """Log de todas as requisições."""
    logger.info(f"Requisição recebida: {request.method} {request.path}")

@app.after_request
def depois_da_requisicao(response):
    """Log das respostas."""
    logger.info(f"Resposta enviada: Status {response.status_code}")
    return response

@app.route('/')
def pagina_inicial():
    """Página inicial com interface para busca de recomendações."""
    try:
        # Parâmetros de paginação
        page = int(request.args.get('page', 1))
        per_page = 20  # usuários por página
        
        # Carregar todos os usuários
        todos_usuarios = carregar_dados_usuarios()
        total_usuarios = len(todos_usuarios)
        
        # Calcular total de páginas
        total_pages = (total_usuarios + per_page - 1) // per_page
        
        # Garantir que a página está dentro dos limites
        page = max(1, min(page, total_pages))
        
        # Selecionar usuários da página atual
        inicio = (page - 1) * per_page
        fim = inicio + per_page
        usuarios_pagina = todos_usuarios[inicio:fim]

        # Calcular range de páginas para paginação
        start_page = max(1, page - 2)
        end_page = min(total_pages, page + 2)
        
        return render_template(
            'index.html', 
            usuarios=usuarios_pagina,
            current_page=page,
            total_pages=total_pages,
            page_size=per_page,
            total_usuarios=total_usuarios,
            page_range=range(start_page, end_page + 1)
        )
                             
    except Exception as e:
        logger.error(f"Erro ao renderizar página inicial: {str(e)}")
        return render_template(
            'index.html', 
            usuarios=[],
            current_page=1,
            total_pages=1,
            page_size=20,
            total_usuarios=0,
            page_range=range(1, 2),
            erro=str(e)
        )

@app.route('/buscar_usuario', methods=['POST'])
def buscar_usuario():
    try:
        usuario_id = request.form.get('usuario_id')
        if not usuario_id:
            return jsonify({"erro": "ID do usuário não fornecido"}), 400
            
        if usuario_id not in modelo.usuario_id_to_index:
            return jsonify({"erro": "Usuário não encontrado"}), 404
            
        # Obter histórico
        historico = modelo.itens_usuario.get(usuario_id, [])
        urls_historico = []
        for idx in list(historico)[-5:]:
            if idx in modelo.index_to_item_id:
                urls_historico.append(modelo.index_to_item_id[idx])
                
        # Gerar recomendações com probabilidades
        recomendacoes = fazer_previsoes(modelo, usuario_id, n_recomendacoes=5)
        
        return jsonify({
            "usuario_id": usuario_id,
            "n_historico": len(historico),
            "ultimos_itens": urls_historico,
            "recomendacoes": recomendacoes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao buscar usuário: {str(e)}")
        return jsonify({"erro": str(e)}), 500

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

@app.route('/api/prever', methods=['POST'])
@tratar_excecoes
@validar_entrada_json(['id_usuario'])
def api_previsoes():
    """Endpoint API para obter recomendações."""
    try:
        dados = request.get_json()
        id_usuario = dados['id_usuario']
        n_recomendacoes = dados.get('n_recomendacoes', 5)
        
        logger.info(f"API: Gerando recomendações para usuário: {id_usuario}")
        
        with mlflow.start_run(run_name=f"api_predicao_{id_usuario}"):
            recomendacoes = fazer_previsoes(modelo, id_usuario, n_recomendacoes)
            
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
        logger.error(f"Erro na API ao gerar recomendações: {str(e)}")
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

@app.teardown_appcontext
def limpar_recursos(error):
    """Limpa recursos ao encerrar a aplicação."""
    try:
        if 'spark' in globals():
            spark.stop()
        gc.collect()
    except Exception as e:
        logger.error(f"Erro ao limpar recursos: {str(e)}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)