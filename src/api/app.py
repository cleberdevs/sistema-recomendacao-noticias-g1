from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_restx import Api, Resource, fields
from src.modelo.recomendador import RecomendadorHibrido
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, explode, count
import gc
from prever import fazer_previsoes, gerar_recomendacoes_cold_start, gerar_recomendacoes_hibridas
import traceback

# Configurar logging
configurar_logging()
logger = get_logger(__name__)

app = Flask(__name__)

# Configurar o Swagger UI para um prefixo específico
api = Api(app, 
    title='API de Recomendações',
    version='1.0',
    description='API para sistema de recomendação híbrido',
    doc='/docs',
    prefix='/api'
)

# Definir namespaces
ns_recomendador = api.namespace('recomendador', description='Operações de recomendação')
ns_sistema = api.namespace('sistema', description='Operações do sistema')

# Definir modelos para documentação
previsao_input = api.model('PrevisaoInput', {
    'id_usuario': fields.String(required=True, description='ID do usuário'),
    'n_recomendacoes': fields.Integer(required=False, description='Número de recomendações desejadas', default=5)
})

recomendacao_output = api.model('RecomendacaoOutput', {
    'recomendacoes': fields.List(fields.String, description='Lista de URLs recomendadas'),
    'metadata': fields.Raw(description='Metadados da recomendação')
})

saude_output = api.model('SaudeOutput', {
    'status': fields.String(description='Status do sistema'),
    'versao_modelo': fields.String(description='Versão do modelo'),
    'timestamp': fields.String(description='Timestamp da verificação'),
    'detalhes': fields.Raw(description='Detalhes adicionais')
})

# Variáveis globais para armazenar dados
timestamps_items = {}
popularidade_items = {}

def calcular_popularidade_items(spark, caminho_treino):
    """
    Calcula a popularidade dos itens usando o arquivo de treino completo.
    """
    try:
        # Carregar dados de treino
        df_treino = spark.read.parquet(caminho_treino)
        
        # Explodir o array de histórico para ter uma linha por interação
        df_interacoes = df_treino.select(
            explode('historico').alias('page')
        )
        
        # Contar ocorrências de cada página
        df_popularidade = df_interacoes.groupBy('page') \
            .agg(count('*').alias('n_interacoes'))
        
        # Coletar resultados
        contagens = {
            row['page']: row['n_interacoes'] 
            for row in df_popularidade.collect()
        }
        
        # Normalizar
        max_contagem = max(contagens.values())
        min_contagem = min(contagens.values())
        
        popularidade_norm = {
            url: (count - min_contagem) / (max_contagem - min_contagem)
            for url, count in contagens.items()
        }
        
        logger.info(f"Calculada popularidade para {len(popularidade_norm)} itens")
        logger.info(f"Contagem máxima: {max_contagem}, mínima: {min_contagem}")
        
        return popularidade_norm
        
    except Exception as e:
        logger.error(f"Erro ao calcular popularidade: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return {}

# Carregar modelo e dados
try:
    logger.info("Inicializando API e carregando modelo...")
    modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    logger.info("Modelo carregado com sucesso")

    spark = SparkSession.builder \
        .appName("RecomendadorAPI") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    logger.info("Carregando dados dos itens...")
    caminho_itens = 'dados_processados/dados_itens_processados.parquet'
    
    dados_itens = spark.read.parquet(caminho_itens) \
        .select('page', 'DataPublicacao')
    
    BATCH_SIZE = 1000
    total_registros = dados_itens.count()
    logger.info(f"Total de registros a processar: {total_registros}")
    
    for offset in range(0, total_registros, BATCH_SIZE):
        batch = dados_itens.limit(BATCH_SIZE).offset(offset).toPandas()
        
        for _, row in batch.iterrows():
            try:
                if pd.notna(row['DataPublicacao']):
                    data = pd.to_datetime(row['DataPublicacao'])
                    if 1970 <= data.year <= 2030:
                        timestamps_items[row['page']] = data.timestamp()
            except:
                continue
        
        if offset % (BATCH_SIZE * 10) == 0:
            logger.info(f"Processados {offset + len(batch)} de {total_registros} registros")

    logger.info("Calculando popularidade dos itens...")
    caminho_treino = 'dados_processados/dados_treino_processados.parquet'
    popularidade_items = calcular_popularidade_items(spark, caminho_treino)

    logger.info(f"Total de itens com timestamp: {len(timestamps_items)}")
    logger.info(f"Total de itens com popularidade: {len(popularidade_items)}")
    
    if timestamps_items:
        data_mais_antiga = datetime.fromtimestamp(min(timestamps_items.values()))
        data_mais_recente = datetime.fromtimestamp(max(timestamps_items.values()))
        logger.info(f"Período coberto: de {data_mais_antiga.date()} até {data_mais_recente.date()}")

    logger.info("Inicialização concluída com sucesso")

    spark.stop()
    gc.collect()

except Exception as e:
    logger.error(f"Erro fatal ao inicializar API: {str(e)}")
    logger.error(f"Traceback completo:\n{traceback.format_exc()}")
    if 'spark' in locals():
        spark.stop()
    raise

@app.route('/')
def pagina_inicial():
    """Página inicial com interface para busca de recomendações."""
    try:
        page = int(request.args.get('page', 1))
        per_page = 20
        
        todos_usuarios = []
        
        usuarios_novos = [
            {'id': f'novo_usuario_{i}', 'n_historico': 0, 'tipo': 'novo'} 
            for i in range(1, 6)
        ]
        todos_usuarios.extend(usuarios_novos)
        
        for usuario_id in modelo.usuario_id_to_index.keys():
            n_historico = len(modelo.itens_usuario.get(usuario_id, []))
            todos_usuarios.append({
                'id': usuario_id,
                'n_historico': n_historico,
                'tipo': 'existente'
            })
            
        total_usuarios = len(todos_usuarios)
        total_pages = (total_usuarios + per_page - 1) // per_page
        
        page = max(1, min(page, total_pages))
        
        inicio = (page - 1) * per_page
        fim = inicio + per_page
        usuarios_pagina = todos_usuarios[inicio:fim]

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
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
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
    """Endpoint para buscar detalhes de um usuário específico."""
    try:
        usuario_id = request.form.get('usuario_id')
        if not usuario_id:
            return jsonify({"erro": "ID do usuário não fornecido"}), 400
            
        logger.info(f"Buscando recomendações para usuário: {usuario_id}")
            
        is_novo = usuario_id.startswith('novo_usuario_')
            
        try:
            recomendacoes = fazer_previsoes(
                modelo, 
                usuario_id, 
                timestamps_items,
                popularidade_items,
                n_recomendacoes=5
            )
            
            if not recomendacoes:
                return jsonify({
                    "erro": "Não foi possível gerar recomendações para este usuário"
                }), 404
            
            logger.info(f"Recomendações geradas: {len(recomendacoes)}")
            
            resposta = {
                "usuario_id": usuario_id,
                "tipo_usuario": "novo" if is_novo else "existente",
                "n_historico": 0 if is_novo else len(modelo.itens_usuario.get(usuario_id, [])),
                "recomendacoes": recomendacoes,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if not is_novo:
                historico = modelo.itens_usuario.get(usuario_id, [])
                urls_historico = []
                for idx in list(historico)[-5:]:
                    if idx in modelo.index_to_item_id:
                        urls_historico.append(modelo.index_to_item_id[idx])
                resposta["ultimos_itens"] = urls_historico
            
            logger.info("Resposta preparada com sucesso")
            return jsonify(resposta), 200
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return jsonify({"erro": str(e)}), 500
        
    except Exception as e:
        logger.error(f"Erro ao buscar usuário: {str(e)}")
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        return jsonify({"erro": str(e)}), 500

@ns_recomendador.route('/prever')
class Previsao(Resource):
    @ns_recomendador.expect(previsao_input)
    @ns_recomendador.response(200, 'Sucesso', recomendacao_output)
    @ns_recomendador.response(400, 'Entrada inválida')
    @ns_recomendador.response(500, 'Erro interno')
    def post(self):
        """Gerar recomendações para um usuário"""
        try:
            dados = request.get_json()
            id_usuario = dados['id_usuario']
            n_recomendacoes = dados.get('n_recomendacoes', 5)
            
            recomendacoes = fazer_previsoes(
                modelo, 
                id_usuario, 
                timestamps_items,
                popularidade_items,
                n_recomendacoes
            )
            
            if not recomendacoes:
                return api.abort(404, "Não foi possível gerar recomendações")
            
            is_novo = id_usuario.startswith('novo_usuario_')
            
            return {
                "recomendacoes": recomendacoes,
                "metadata": {
                    "usuario": id_usuario,
                    "tipo_usuario": "novo" if is_novo else "existente",
                    "quantidade": len(recomendacoes),
                    "timestamp": datetime.now().isoformat(),
                    "detalhes_modelo": {
                        "cold_start": {
                            "peso_popularidade": 0.7,
                            "peso_recencia": 0.3
                        },
                        "hibrido": {
                            "peso_modelo": 0.6,
                            "peso_popularidade": 0.25,
                            "peso_recencia": 0.15
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na API ao gerar recomendações: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return api.abort(500, f"Erro ao gerar recomendações: {str(e)}")

@ns_sistema.route('/saude')
class Saude(Resource):
    @ns_sistema.response(200, 'Sucesso', saude_output)
    @ns_sistema.response(500, 'Erro interno')
    def get(self):
        """Verificar saúde do sistema"""
        try:
            if modelo is None:
                return api.abort(500, "Modelo não carregado")
                
            return {
                "status": "saudavel",
                "versao_modelo": os.getenv('MODEL_VERSION', 'v1'),
                "timestamp": datetime.now().isoformat(),
                "detalhes": {
                    "tipo_modelo": "híbrido",
                    "suporta_cold_start": True,
                    "componentes": ["neural", "popularidade", "recência"],
                    "itens_com_timestamp": len(timestamps_items),
                    "itens_com_popularidade": len(popularidade_items)
                }
            }
        except Exception as e:
            logger.error(f"Erro na verificação de saúde: {str(e)}")
            return api.abort(500, f"Erro na verificação de saúde: {str(e)}")

@app.route('/templates/<path:path>')
def send_template(path):
    return send_from_directory('templates', path)

@app.errorhandler(404)
def pagina_nao_encontrada(e):
    if request.path.startswith('/api/'):
        return jsonify({"erro": "Rota da API não encontrada"}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def erro_servidor(e):
    if request.path.startswith('/api/'):
        return jsonify({"erro": "Erro interno do servidor"}), 500
    return render_template('500.html'), 500

# Configurações adicionais do Swagger
@api.errorhandler(Exception)
def handle_error(error):
    """Manipulador global de erros para a API"""
    message = str(error)
    status_code = getattr(error, 'code', 500)
    
    if status_code == 500:
        logger.error(f"Erro interno: {message}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    return {
        'mensagem': message,
        'status_code': status_code
    }, status_code

# Configurações para CORS
@app.after_request
def after_request(response):
    """Adiciona headers para permitir CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.teardown_appcontext
def limpar_recursos(error):
    """Limpa recursos ao encerrar a aplicação."""
    try:
        gc.collect()
    except Exception as e:
        logger.error(f"Erro ao limpar recursos: {str(e)}")

if __name__ == '__main__':
    try:
        # Configurações do servidor
        port = int(os.getenv('PORT', 8000))
        debug = os.getenv('FLASK_ENV', 'development') == 'development'
        
        # Log das configurações
        logger.info(f"Iniciando servidor na porta {port}")
        logger.info(f"Modo debug: {debug}")
        logger.info(f"Documentação Swagger disponível em: http://localhost:{port}/docs")
        
        # Iniciar servidor
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=debug
        )
    except Exception as e:
        logger.critical(f"Erro fatal ao iniciar servidor: {str(e)}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}")
        raise