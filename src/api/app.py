from flask import Flask, request, jsonify, render_template
from src.modelo.recomendador import RecomendadorHibrido
from src.utils.helpers import tratar_excecoes, validar_entrada_json
from src.config.logging_config import get_logger, configurar_logging
import mlflow
from datetime import datetime
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year
import gc
from prever import fazer_previsoes, gerar_recomendacoes_cold_start, gerar_recomendacoes_hibridas
import traceback

# Configurar logging
configurar_logging()
logger = get_logger(__name__)

app = Flask(__name__)

# Variável global para armazenar timestamps
timestamps_items = {}

# Carregar modelo e dados
try:
    # 1. Carregar o modelo treinado (não será modificado)
    logger.info("Inicializando API e carregando modelo...")
    modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    logger.info("Modelo carregado com sucesso")

    # 2. Inicializar Spark com configurações adequadas
    spark = SparkSession.builder \
        .appName("RecomendadorAPI") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    # 3. Carregar dados dos itens apenas para obter as datas
    logger.info("Carregando dados dos itens...")
    caminho_itens = 'dados/processados/dados_itens_processados.parquet'
    
    dados_itens = spark.read.parquet(caminho_itens) \
        .select('page', 'DataPublicacao')
    
    # 4. Processar as datas em lotes
    BATCH_SIZE = 1000
    total_registros = dados_itens.count()
    logger.info(f"Total de registros a processar: {total_registros}")
    
    for offset in range(0, total_registros, BATCH_SIZE):
        batch = dados_itens.limit(BATCH_SIZE).offset(offset).toPandas()
        
        for _, row in batch.iterrows():
            try:
                if pd.isna(row['DataPublicacao']):
                    timestamps_items[row['page']] = datetime(2024, 1, 1).timestamp()
                    continue
                    
                data = pd.to_datetime(row['DataPublicacao'])
                if data.year < 1970 or data.year > 2030:
                    timestamps_items[row['page']] = datetime(2024, 1, 1).timestamp()
                else:
                    timestamps_items[row['page']] = data.timestamp()
            except:
                timestamps_items[row['page']] = datetime(2024, 1, 1).timestamp()
        
        if offset % (BATCH_SIZE * 10) == 0:
            logger.info(f"Processados {offset + len(batch)} de {total_registros} registros")

    # 5. Log das estatísticas
    logger.info(f"Total de itens com timestamp: {len(timestamps_items)}")
    
    # Verificar distribuição temporal
    timestamps_validos = [ts for ts in timestamps_items.values() 
                         if ts != datetime(2024, 1, 1).timestamp()]
    if timestamps_validos:
        data_mais_antiga = datetime.fromtimestamp(min(timestamps_validos))
        data_mais_recente = datetime.fromtimestamp(max(timestamps_validos))
        logger.info(f"Período coberto: de {data_mais_antiga.date()} até {data_mais_recente.date()}")

    logger.info("Inicialização concluída com sucesso")

    # 6. Limpar recursos do Spark
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
        # Parâmetros de paginação
        page = int(request.args.get('page', 1))
        per_page = 20
        
        # Carregar todos os usuários
        todos_usuarios = []
        
        # Adicionar alguns usuários novos (sem histórico)
        usuarios_novos = [
            {'id': f'novo_usuario_{i}', 'n_historico': 0, 'tipo': 'novo'} 
            for i in range(1, 6)  # 5 usuários novos
        ]
        todos_usuarios.extend(usuarios_novos)
        
        # Adicionar usuários existentes
        for usuario_id in modelo.usuario_id_to_index.keys():
            n_historico = len(modelo.itens_usuario.get(usuario_id, []))
            todos_usuarios.append({
                'id': usuario_id,
                'n_historico': n_historico,
                'tipo': 'existente'
            })
            
        total_usuarios = len(todos_usuarios)
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
            
        # Verificar se é um usuário novo (sem histórico)
        is_novo = usuario_id.startswith('novo_usuario_')
            
        try:
            # Gerar recomendações passando o dicionário de timestamps
            recomendacoes = fazer_previsoes(modelo, usuario_id, timestamps_items, n_recomendacoes=5)
            
            if not recomendacoes:
                return jsonify({
                    "erro": "Não foi possível gerar recomendações para este usuário"
                }), 404
            
            logger.info(f"Recomendações geradas: {len(recomendacoes)}")
            
            # Validar formato das recomendações
            for rec in recomendacoes:
                if 'url' not in rec or 'score' not in rec:
                    logger.error(f"Formato inválido de recomendação: {rec}")
                    return jsonify({
                        "erro": "Erro no formato das recomendações"
                    }), 500
            
            # Preparar resposta
            resposta = {
                "usuario_id": usuario_id,
                "tipo_usuario": "novo" if is_novo else "existente",
                "n_historico": 0 if is_novo else len(modelo.itens_usuario.get(usuario_id, [])),
                "recomendacoes": recomendacoes,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Adicionar histórico apenas se usuário existente
            if not is_novo:
                historico = modelo.itens_usuario.get(usuario_id, [])
                urls_historico = []
                for idx in list(historico)[-5:]:  # Últimos 5 itens
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
        
        # Gerar recomendações passando o dicionário de timestamps
        recomendacoes = fazer_previsoes(modelo, id_usuario, timestamps_items, n_recomendacoes)
        
        if not recomendacoes:
            return jsonify({
                "erro": "Não foi possível gerar recomendações"
            }), 404
        
        # Determinar tipo de usuário
        is_novo = id_usuario.startswith('novo_usuario_')
        
        return jsonify({
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
        }), 200
        
    except Exception as e:
        logger.error(f"Erro na API ao gerar recomendações: {str(e)}")
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
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
            "timestamp": datetime.now().isoformat(),
            "detalhes": {
                "tipo_modelo": "híbrido",
                "suporta_cold_start": True,
                "componentes": ["neural", "popularidade", "recência"],
                "itens_com_timestamp": len(timestamps_items)
            }
        }), 200
    except Exception as e:
        logger.error(f"Erro na verificação de saúde: {str(e)}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

@app.errorhandler(404)
def nao_encontrado(erro):
    """Handler para rotas não encontradas."""
    logger.warning(f"Rota não encontrada: {request.path}")
    return jsonify({"erro": "Rota não encontrada"}), 404

@app.errorhandler(500)
def erro_interno(erro):
    """Handler para erros internos."""
    logger.error(f"Erro interno do servidor: {str(erro)}")
    logger.error(f"Traceback completo:\n{traceback.format_exc()}")
    return jsonify({
        "erro": "Erro interno do servidor",
        "detalhes": str(erro)
    }), 500

@app.teardown_appcontext
def limpar_recursos(error):
    """Limpa recursos ao encerrar a aplicação."""
    try:
        gc.collect()
    except Exception as e:
        logger.error(f"Erro ao limpar recursos: {str(e)}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)