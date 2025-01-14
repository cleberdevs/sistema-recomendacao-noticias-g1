'''import os
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
        raise'''
import os
import glob
import shutil
import logging
from pathlib import Path
from typing import List, Any, Dict
from functools import wraps
from flask import jsonify, request

logger = logging.getLogger(__name__)

def carregar_arquivos_dados(padrao: str) -> List[str]:
    """
    Carrega arquivos que correspondem ao padrão especificado.
    
    Args:
        padrao: Padrão glob para busca de arquivos
        
    Returns:
        Lista de caminhos dos arquivos encontrados
    """
    # Converter para Path para manipulação mais segura
    padrao_path = Path(padrao)
    
    # Resolver caminho completo
    caminho_base = padrao_path.parent
    padrao_arquivo = padrao_path.name
    
    # Verificar se o diretório existe
    if not caminho_base.exists():
        logger.error(f"Diretório não encontrado: {caminho_base}")
        return []
        
    # Buscar arquivos
    arquivos = list(caminho_base.glob(padrao_arquivo))
    
    if not arquivos:
        logger.warning(f"Nenhum arquivo encontrado com o padrão: {padrao}")
    else:
        logger.info(f"Encontrados {len(arquivos)} arquivos")
        for arquivo in arquivos:
            logger.debug(f"Arquivo encontrado: {arquivo}")
            
            # Verificar tamanho do arquivo
            tamanho = arquivo.stat().st_size / (1024 * 1024)  # MB
            logger.debug(f"Tamanho: {tamanho:.2f} MB")
    
    return [str(arquivo.absolute()) for arquivo in arquivos]

def criar_diretorio_se_nao_existe(caminho: str) -> None:
    """
    Cria um diretório se ele não existir.
    
    Args:
        caminho: Caminho do diretório a ser criado
    """
    diretorio = Path(caminho)
    if not diretorio.exists():
        logger.info(f"Criando diretório: {caminho}")
        diretorio.mkdir(parents=True, exist_ok=True)
    else:
        logger.debug(f"Diretório já existe: {caminho}")

def verificar_espaco_disco(caminho: str, espaco_minimo_gb: float = 10.0) -> bool:
    """
    Verifica se há espaço em disco suficiente.
    
    Args:
        caminho: Caminho para verificar
        espaco_minimo_gb: Espaço mínimo necessário em GB
        
    Returns:
        bool: True se há espaço suficiente
    """
    try:
        total, usado, livre = shutil.disk_usage(caminho)
        livre_gb = livre / (1024 * 1024 * 1024)  # Converter para GB
        
        logger.info(f"Espaço livre em disco: {livre_gb:.2f} GB")
        
        if livre_gb < espaco_minimo_gb:
            logger.warning(
                f"Espaço em disco insuficiente. "
                f"Disponível: {livre_gb:.2f} GB, "
                f"Mínimo necessário: {espaco_minimo_gb} GB"
            )
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Erro ao verificar espaço em disco: {str(e)}")
        return False

def limpar_arquivos_temporarios(diretorio: str, padrao: str = "*") -> None:
    """
    Remove arquivos temporários.
    
    Args:
        diretorio: Diretório onde procurar
        padrao: Padrão de arquivos a remover
    """
    try:
        dir_path = Path(diretorio)
        if not dir_path.exists():
            return
            
        arquivos = list(dir_path.glob(padrao))
        for arquivo in arquivos:
            if arquivo.is_file():
                arquivo.unlink()
                logger.debug(f"Arquivo removido: {arquivo}")
                
        logger.info(f"{len(arquivos)} arquivos temporários removidos")
        
    except Exception as e:
        logger.error(f"Erro ao limpar arquivos temporários: {str(e)}")

def tratar_excecoes(f):
    """
    Decorator para tratamento uniforme de exceções na API.
    
    Args:
        f: Função a ser decorada
        
    Returns:
        Função decorada
    """
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