import os
import glob
import pandas as pd
from typing import List, Any, Dict
from functools import wraps
from flask import jsonify, request
from src.config.logging_config import get_logger

logger = get_logger(__name__)

def carregar_arquivos_dados(padrao: str) -> List[str]:
    """
    Carrega arquivos que correspondem ao padrão especificado.
    
    Args:
        padrao: Padrão glob para busca de arquivos
        
    Returns:
        Lista de caminhos dos arquivos encontrados
    """
    logger.info(f"Buscando arquivos com padrão: {padrao}")
    arquivos = glob.glob(padrao)
    
    if not arquivos:
        logger.warning(f"Nenhum arquivo encontrado com o padrão: {padrao}")
    else:
        logger.info(f"Encontrados {len(arquivos)} arquivos")
        logger.debug(f"Arquivos: {arquivos}")
        
    return arquivos

def validar_entrada_json(campos_obrigatorios: List[str]):
    """
    Decorator para validar entrada JSON da API.
    
    Args:
        campos_obrigatorios: Lista de campos que devem estar presentes no JSON
    """
    def decorador(f):
        @wraps(f)
        def funcao_decorada(*args, **kwargs):
            try:
                dados = request.get_json()
                if not dados:
                    erro_msg = "Dados JSON não fornecidos"
                    logger.error(erro_msg)
                    return jsonify({
                        "erro": erro_msg,
                        "detalhes": "O corpo da requisição deve ser JSON válido"
                    }), 400
                
                campos_faltantes = [
                    campo for campo in campos_obrigatorios 
                    if campo not in dados
                ]
                
                if campos_faltantes:
                    erro_msg = f"Campos obrigatórios faltando: {campos_faltantes}"
                    logger.error(erro_msg)
                    return jsonify({
                        "erro": "Campos obrigatórios faltando",
                        "campos_faltantes": campos_faltantes
                    }), 400
                
                # Validar tipos de dados
                if 'id_usuario' in dados and not isinstance(dados['id_usuario'], str):
                    return jsonify({
                        "erro": "Tipo inválido",
                        "detalhes": "id_usuario deve ser uma string"
                    }), 400
                
                if 'n_recomendacoes' in dados and not isinstance(dados['n_recomendacoes'], int):
                    return jsonify({
                        "erro": "Tipo inválido",
                        "detalhes": "n_recomendacoes deve ser um inteiro"
                    }), 400
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Erro na validação de entrada: {str(e)}")
                return jsonify({
                    "erro": "Erro na validação de entrada",
                    "detalhes": str(e)
                }), 400
                
        return funcao_decorada
    return decorador

def tratar_excecoes(f):
    """Decorator para tratamento uniforme de exceções na API."""
    @wraps(f)
    def funcao_decorada(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erro na execução: {str(e)}")
            return jsonify({
                "erro": "Erro interno",
                "tipo": type(e).__name__,
                "detalhes": str(e)
            }), 500
    return funcao_decorada

def validar_tipo_dados(valor: Any, tipo_esperado: type, nome_campo: str) -> Any:
    """
    Valida o tipo de um valor.
    
    Args:
        valor: Valor a ser validado
        tipo_esperado: Tipo esperado
        nome_campo: Nome do campo para mensagem de erro
        
    Returns:
        O próprio valor se for válido
        
    Raises:
        ValueError: Se o tipo for inválido
    """
    if not isinstance(valor, tipo_esperado):
        erro_msg = f"Campo '{nome_campo}' deve ser do tipo {tipo_esperado.__name__}"
        logger.error(erro_msg)
        raise ValueError(erro_msg)
    return valor

def criar_diretorio_se_nao_existe(diretorio: str) -> None:
    """Cria um diretório se ele não existir."""
    if not os.path.exists(diretorio):
        logger.info(f"Criando diretório: {diretorio}")
        os.makedirs(diretorio)

def salvar_metricas(metricas: Dict[str, float], caminho: str) -> None:
    """
    Salva métricas em um arquivo JSON.
    
    Args:
        metricas: Dicionário com as métricas
        caminho: Caminho do arquivo
    """
    try:
        import json
        with open(caminho, 'w') as f:
            json.dump(metricas, f, indent=4)
        logger.info(f"Métricas salvas em: {caminho}")
    except Exception as e:
        logger.error(f"Erro ao salvar métricas: {str(e)}")
        raise