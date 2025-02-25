import os
import glob
import shutil
import zipfile
import logging
from pathlib import Path
from typing import List, Any, Dict
from functools import wraps
from flask import jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def descompacta_csvs(zip_path, diretorio_destino, tipo_arquivo='treino'):
    """
    Descompacta arquivos CSV específicos de um arquivo ZIP para um diretório destino.

    Args:
        zip_path: Caminho para o arquivo ZIP
        diretorio_destino: Diretório onde os arquivos serão extraídos
        tipo_arquivo: 'treino' para arquivos de treino ou 'itens' para arquivos de itens

    Returns:
        Lista com caminhos de todos os arquivos CSV extraídos
    """
    arquivos_extraidos = []

    # Configurar padrões de busca com base no tipo de arquivo
    if tipo_arquivo == 'treino':
        pasta_dentro_zip = 'files/treino'
        padrao_arquivo = 'treino_parte*.csv'
    elif tipo_arquivo == 'itens':
        pasta_dentro_zip = 'itens/itens'
        padrao_arquivo = 'itens-parte*.csv'
    else:
        logger.error(f"Tipo de arquivo desconhecido: {tipo_arquivo}")
        return []

    logger.info(f"Descompactando arquivos {tipo_arquivo} do ZIP para {diretorio_destino}")

    try:
        # Garantir que o diretório de destino exista
        os.makedirs(diretorio_destino, exist_ok=True)

        # Verificar se o arquivo ZIP existe
        if not os.path.exists(zip_path):
            logger.error(f"Arquivo ZIP não encontrado: {zip_path}")
            # Tentar encontrar o ZIP em outros caminhos comuns
            possivel_zip = glob.glob("challenge-webmedia-e-globo-2023.zip") or \
                           glob.glob("*/challenge-webmedia-e-globo-2023.zip") or \
                           glob.glob("/app/challenge-webmedia-e-globo-2023.zip")

            if possivel_zip:
                zip_path = possivel_zip[0]
                logger.info(f"ZIP encontrado em caminho alternativo: {zip_path}")
            else:
                logger.error("Arquivo ZIP não encontrado em nenhum caminho")
                return []

        # Abrir o arquivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Listar todo o conteúdo do ZIP para debug
            todos_arquivos = zip_ref.namelist()
            logger.info(f"Conteúdo do ZIP (primeiros 20 arquivos): {todos_arquivos[:20]}")

            # Filtrar apenas os arquivos que nos interessam
            arquivos_alvo = [f for f in todos_arquivos if
                             (pasta_dentro_zip in f.replace('\\', '/') and
                              f.replace('\\', '/').endswith('.csv') and
                              (('treino_parte' in f and tipo_arquivo == 'treino') or
                               ('itens-parte' in f and tipo_arquivo == 'itens')))]

            logger.info(f"Arquivos {tipo_arquivo} encontrados no ZIP: {arquivos_alvo}")

            if not arquivos_alvo:
                # Tentar um padrão mais flexível
                logger.warning(
                    f"Nenhum arquivo {tipo_arquivo} encontrado com o padrão específico. Tentando padrão mais flexível...")
                arquivos_alvo = [f for f in todos_arquivos if
                                 (f.replace('\\', '/').endswith('.csv') and
                                  ((tipo_arquivo == 'treino' and 'treino' in f.lower()) or
                                   (tipo_arquivo == 'itens' and ('item' in f.lower() or 'iten' in f.lower()))))]

                logger.info(f"Arquivos encontrados com padrão flexível: {arquivos_alvo}")

            # Extrair cada arquivo para o diretório de destino
            for arquivo in arquivos_alvo:
                # Obter apenas o nome do arquivo sem o caminho
                nome_arquivo = os.path.basename(arquivo)
                caminho_destino = os.path.join(diretorio_destino, nome_arquivo)

                # Extrair o arquivo
                try:
                    # Ler o conteúdo do arquivo do ZIP
                    conteudo = zip_ref.read(arquivo)

                    # Escrever no destino
                    with open(caminho_destino, 'wb') as f_destino:
                        f_destino.write(conteudo)

                    logger.info(f"Arquivo extraído: {caminho_destino}")
                    arquivos_extraidos.append(caminho_destino)
                except Exception as e:
                    logger.error(f"Erro ao extrair arquivo {arquivo}: {str(e)}")

    except Exception as e:
        logger.error(f"Erro ao descompactar o arquivo ZIP: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    # Verificar se os arquivos foram extraídos
    if not arquivos_extraidos:
        logger.warning(f"Nenhum arquivo foi extraído do ZIP para {diretorio_destino}")
    else:
        logger.info(f"Total de {len(arquivos_extraidos)} arquivos extraídos para {diretorio_destino}")

    return arquivos_extraidos


def carregar_todos_csvs(diretorio):
    """
    Carrega todos os arquivos CSV de um diretório.
    Mantida para compatibilidade.
    """
    try:
        if not os.path.exists(diretorio):
            logger.warning(f"Diretório não existe: {diretorio}")
            return []

        arquivos = glob.glob(os.path.join(diretorio, "*.csv"))
        logger.info(f"Encontrados {len(arquivos)} arquivos CSV em {diretorio}")
        return arquivos
    except Exception as e:
        logger.error(f"Erro ao carregar arquivos CSV: {str(e)}")
        return []


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