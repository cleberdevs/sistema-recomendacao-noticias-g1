import os
import glob
import pandas as pd
from typing import List
from functools import wraps
from flask import jsonify, request

def carregar_arquivos_dados(padrao: str) -> List[str]:
    return glob.glob(padrao)

def combinar_dataframes(arquivos: List[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

def criar_diretorio_se_nao_existe(diretorio: str):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

def validar_entrada_json(campos_obrigatorios):
    def decorador(f):
        @wraps(f)
        def funcao_decorada(*args, **kwargs):
            dados = request.get_json()
            if not dados:
                return jsonify({"erro": "Dados JSON não fornecidos"}), 400
                
            campos_faltantes = [campo for campo in campos_obrigatorios if campo not in dados]
            if campos_faltantes:
                return jsonify({
                    "erro": f"Campos obrigatórios faltando: {', '.join(campos_faltantes)}"
                }), 400
                
            return f(*args, **kwargs)
        return funcao_decorada
    return decorador

def tratar_excecoes(f):
    @wraps(f)
    def funcao_decorada(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"erro": str(e)}), 500
    return funcao_decorada
