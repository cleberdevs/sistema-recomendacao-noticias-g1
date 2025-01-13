'''import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Tuple, List
import ast
import traceback
import gc
import json
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None
        self.chunk_size = 10000
        self.checkpoint_dir = Path('dados/processados/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Inicializando PreProcessadorDados")

    def _verificar_arquivos(self, arquivos: List[str]) -> None:
        """Verifica o tamanho e existência dos arquivos."""
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo) / (1024 * 1024)  # MB
                logger.info(f"Arquivo {arquivo}: {tamanho:.2f} MB")
            else:
                logger.warning(f"Arquivo não encontrado: {arquivo}")

    def _verificar_e_renomear_colunas(self, df: pd.DataFrame, mapeamento: dict) -> pd.DataFrame:
        """
        Verifica se as colunas existem e renomeia.
        
        Args:
            df: DataFrame a ser processado
            mapeamento: Dicionário com o mapeamento de colunas
            
        Returns:
            DataFrame com colunas renomeadas
        """
        # Verificar se as colunas originais existem
        colunas_faltantes = [col for col in mapeamento.keys() if col not in df.columns]
        if colunas_faltantes:
            erro_msg = f"Colunas originais faltantes: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)
        
        return df.rename(columns=mapeamento)

    def _salvar_checkpoint(self, dados: pd.DataFrame, nome: str, indice: int = None) -> str:
        """
        Salva um checkpoint dos dados processados.
        
        Args:
            dados: DataFrame a ser salvo
            nome: Nome base do arquivo
            indice: Índice do chunk (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        try:
            # Criar nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if indice is not None:
                arquivo = self.checkpoint_dir / f"{nome}_chunk_{indice}_{timestamp}.csv"
            else:
                arquivo = self.checkpoint_dir / f"{nome}_{timestamp}.csv"

            # Converter listas para strings antes de salvar
            dados_para_salvar = dados.copy()
            if 'historico' in dados_para_salvar.columns:
                dados_para_salvar['historico'] = dados_para_salvar['historico'].apply(json.dumps)
            if 'historicoTimestamp' in dados_para_salvar.columns:
                dados_para_salvar['historicoTimestamp'] = dados_para_salvar['historicoTimestamp'].apply(json.dumps)

            # Salvar dados
            dados_para_salvar.to_csv(arquivo, index=False)
            logger.info(f"Checkpoint salvo: {arquivo}")
            return str(arquivo)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            raise

    def _carregar_checkpoints(self, padrao: str) -> pd.DataFrame:
        """
        Carrega checkpoints salvos.
        
        Args:
            padrao: Padrão do nome dos arquivos a serem carregados
            
        Returns:
            DataFrame combinado dos checkpoints
        """
        try:
            arquivos = list(self.checkpoint_dir.glob(padrao))
            if not arquivos:
                return None

            dfs = []
            for arquivo in arquivos:
                logger.info(f"Carregando checkpoint: {arquivo}")
                df = pd.read_csv(arquivo)
                
                # Converter strings de volta para listas
                if 'historico' in df.columns:
                    df['historico'] = df['historico'].apply(json.loads)
                if 'historicoTimestamp' in df.columns:
                    df['historicoTimestamp'] = df['historicoTimestamp'].apply(json.loads)
                
                dfs.append(df)

            return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoints: {str(e)}")
            raise

    def _limpar_checkpoints(self, padrao: str = None):
        """Limpa arquivos de checkpoint."""
        try:
            if padrao:
                arquivos = list(self.checkpoint_dir.glob(padrao))
            else:
                arquivos = list(self.checkpoint_dir.glob('*'))

            for arquivo in arquivos:
                arquivo.unlink()
            logger.info("Checkpoints removidos")

        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")

    def processar_dados_treino(self, arquivos_treino: List[str], 
                             arquivos_itens: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa os dados de treino e itens usando chunks e salvamento intermediário.
        """
        logger.info("Iniciando processamento dos dados")
        
        # Verificar arquivos
        self._verificar_arquivos(arquivos_treino)
        self._verificar_arquivos(arquivos_itens)
        
        try:
            # Limpar checkpoints antigos
            self._limpar_checkpoints()
            
            # Processar dados de treino em chunks
            logger.info("Processando dados de treino")
            checkpoint_files_treino = []
            
            for arquivo in arquivos_treino:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de treino: {arquivo}")
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de treino")
                        logger.info(f"Colunas disponíveis no chunk de treino: {chunk.columns.tolist()}")
                        
                        # Processar chunk
                        chunk = self._verificar_e_renomear_colunas(
                            chunk,
                            {
                                'history': 'historico',
                                'timestampHistory': 'historicoTimestamp',
                                'userId': 'idUsuario'
                            }
                        )
                        
                        chunk['historico'] = chunk['historico'].apply(self._processar_historico)
                        chunk['historicoTimestamp'] = chunk['historicoTimestamp'].apply(
                            self._processar_historico
                        )
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'treino', i)
                        checkpoint_files_treino.append(arquivo_checkpoint)
                        
                        gc.collect()

            # Carregar e combinar checkpoints de treino
            logger.info("Combinando dados de treino")
            dados_treino = self._carregar_checkpoints('treino_chunk_*.csv')
            if dados_treino is None:
                raise ValueError("Falha ao processar dados de treino")

            # Processar dados dos itens
            logger.info("Processando dados de itens")
            checkpoint_files_itens = []
            
            for arquivo in arquivos_itens:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de itens: {arquivo}")
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de itens")
                        logger.info(f"Colunas disponíveis no chunk de itens: {chunk.columns.tolist()}")
                        
                        # Processar chunk
                        chunk = self._verificar_e_renomear_colunas(
                            chunk,
                            {
                                'Page': 'Pagina',
                                'Title': 'Titulo',
                                'Body': 'Corpo',
                                'Issued': 'DataPublicacao'
                            }
                        )
                        
                        # Converter timestamp
                        chunk['DataPublicacao'] = pd.to_datetime(chunk['DataPublicacao'])
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'itens', i)
                        checkpoint_files_itens.append(arquivo_checkpoint)
                        
                        gc.collect()

            # Carregar e combinar checkpoints de itens
            logger.info("Combinando dados de itens")
            dados_itens = self._carregar_checkpoints('itens_chunk_*.csv')
            if dados_itens is None:
                raise ValueError("Falha ao processar dados de itens")

            # Validar dados finais
            self._validar_colunas_treino(dados_treino)
            self._validar_colunas_itens(dados_itens)

            # Salvar resultado final
            logger.info("Salvando dados processados finais")
            self._salvar_checkpoint(dados_treino, 'treino_final')
            self._salvar_checkpoint(dados_itens, 'itens_final')

            # Limpar checkpoints intermediários
            self._limpar_checkpoints('*chunk*.csv')

            logger.info("Processamento dos dados concluído com sucesso")
            return dados_treino, dados_itens

        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        finally:
            # Garantir que a memória seja liberada
            gc.collect()

    def _validar_colunas_treino(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de treino."""
        colunas_necessarias = ['historico', 'historicoTimestamp', 'idUsuario']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de treino: {colunas_faltantes}"
            logger.error(erro_msg)
            raise ValueError(erro_msg)

    def _validar_colunas_itens(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de itens."""
        colunas_necessarias = ['Pagina', 'Titulo', 'Corpo', 'DataPublicacao']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de itens: {colunas_faltantes}"
            logger.error(erro_msg)
            raise ValueError(erro_msg)

    def _processar_historico(self, valor: str) -> list:
        """
        Processa strings de histórico para listas.
        
        Args:
            valor: String contendo a lista a ser processada
            
        Returns:
            list: Lista processada
        """
        try:
            if isinstance(valor, list):
                return valor
            if isinstance(valor, str):
                try:
                    return ast.literal_eval(valor)
                except:
                    # Se falhar, tentar limpar a string e processar
                    valor_limpo = valor.strip('[]').replace(' ', '').split(',')
                    return [v.strip("'") for v in valor_limpo if v]
            return []
        except Exception as e:
            logger.error(f"Erro ao processar histórico: {str(e)}")
            logger.error(f"Valor problemático: {valor}")
            return []

    def preparar_features_texto(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """Prepara features de texto combinando título e corpo."""
        logger.info("Preparando features de texto")
        try:
            dados_itens['conteudo_texto'] = dados_itens['Titulo'].fillna('') + ' ' + \
                                          dados_itens['Corpo'].fillna('')
            return dados_itens
        except Exception as e:
            logger.error(f"Erro ao preparar features de texto: {str(e)}")
            raise

    def validar_dados(self, dados_treino: pd.DataFrame, 
                     dados_itens: pd.DataFrame) -> bool:
        """
        Realiza validações adicionais nos dados.
        
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Iniciando validação dos dados")
        try:
            # Verificar valores nulos
            nulos_treino = dados_treino.isnull().sum()
            nulos_itens = dados_itens.isnull().sum()
            
            if nulos_treino.any():
                logger.warning("Valores nulos encontrados nos dados de treino:")
                for coluna, quantidade in nulos_treino[nulos_treino > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")
            
            if nulos_itens.any():
                logger.warning("Valores nulos encontrados nos dados de itens:")
                for coluna, quantidade in nulos_itens[nulos_itens > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")

            # Verificar consistência dos dados
            self._verificar_consistencia_dados(dados_treino, dados_itens)
            
            logger.info("Validação dos dados concluída")
            return True

        except Exception as e:
            logger.error(f"Erro na validação dos dados: {str(e)}")
            return False

    def _verificar_consistencia_dados(self, dados_treino: pd.DataFrame, 
                                    dados_itens: pd.DataFrame) -> None:
        """Verifica a consistência entre dados de treino e itens."""
        try:
            # Verificar se todos os itens do histórico existem nos dados de itens
            todos_itens = set(dados_itens['Pagina'])
            itens_historico = set([
                item for historico in dados_treino['historico'] 
                for item in historico
            ])
            
            itens_faltantes = itens_historico - todos_itens
            if itens_faltantes:
                logger.warning(
                    f"Existem {len(itens_faltantes)} itens no histórico que não existem nos dados de itens"
                )
                
        except Exception as e:
            logger.error(f"Erro ao verificar consistência dos dados: {str(e)}")
            raise

    def mostrar_info_dados(self, dados_treino: pd.DataFrame, 
                          dados_itens: pd.DataFrame) -> None:
        """Mostra informações detalhadas sobre os dados."""
        logger.info("Exibindo informações dos dados")
        try:
            print("\nInformações dos dados de treino:")
            print(f"Número de registros: {len(dados_treino)}")
            print(f"Número de usuários únicos: {dados_treino['idUsuario'].nunique()}")
            print(f"Média de itens por usuário: {dados_treino['historico'].apply(len).mean():.2f}")
            print("\nAmostra dos dados de treino:")
            print(dados_treino.head())
            
            print("\nInformações dos dados de itens:")
            print(f"Número de itens: {len(dados_itens)}")
            print(f"Período dos dados: {dados_itens['DataPublicacao'].min()} até {dados_itens['DataPublicacao'].max()}")
            print("\nAmostra dos dados de itens:")
            print(dados_itens.head())
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações dos dados: {str(e)}")
            raise'''



'''import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Tuple, List
import ast
import traceback
import gc
import json
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None
        self.chunk_size = 10000
        self.checkpoint_dir = Path('dados/processados/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Inicializando PreProcessadorDados")

    def _verificar_arquivos(self, arquivos: List[str]) -> None:
        """Verifica o tamanho e existência dos arquivos."""
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo) / (1024 * 1024)  # MB
                logger.info(f"Arquivo {arquivo}: {tamanho:.2f} MB")
            else:
                logger.warning(f"Arquivo não encontrado: {arquivo}")

    def _determinar_mapeamento_colunas(self, colunas_disponiveis: List[str]) -> dict:
        """
        Determina o mapeamento correto de colunas baseado nas colunas disponíveis.
        """
        # Possíveis nomes para cada coluna
        mapeamentos_possiveis = {
            'Pagina': ['Page', 'page', 'PAGE', 'url', 'URL', 'link'],
            'Titulo': ['Title', 'title', 'TITLE', 'titulo', 'Título'],
            'Corpo': ['Body', 'body', 'BODY', 'texto', 'content', 'Content'],
            'DataPublicacao': ['Issued', 'issued', 'ISSUED', 'data', 'date', 'Data', 'published_date']
        }

        mapeamento_final = {}
        colunas_disponiveis = set(colunas_disponiveis)
        
        for coluna_final, possiveis_nomes in mapeamentos_possiveis.items():
            for nome in possiveis_nomes:
                if nome in colunas_disponiveis:
                    mapeamento_final[nome] = coluna_final
                    break
        
        logger.info(f"Mapeamento de colunas determinado: {mapeamento_final}")
        return mapeamento_final

    def _verificar_e_renomear_colunas(self, df: pd.DataFrame, mapeamento: dict) -> pd.DataFrame:
        """
        Verifica se as colunas existem e renomeia.
        
        Args:
            df: DataFrame a ser processado
            mapeamento: Dicionário com o mapeamento de colunas
            
        Returns:
            DataFrame com colunas renomeadas
        """
        logger.info(f"Verificando e renomeando colunas. Colunas disponíveis: {df.columns.tolist()}")
        logger.info(f"Mapeamento a ser aplicado: {mapeamento}")
        
        # Verificar se as colunas originais existem
        colunas_faltantes = [col for col in mapeamento.keys() if col not in df.columns]
        if colunas_faltantes:
            erro_msg = f"Colunas originais faltantes: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)
        
        return df.rename(columns=mapeamento)

    def _salvar_checkpoint(self, dados: pd.DataFrame, nome: str, indice: int = None) -> str:
        """
        Salva um checkpoint dos dados processados.
        
        Args:
            dados: DataFrame a ser salvo
            nome: Nome base do arquivo
            indice: Índice do chunk (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        try:
            # Criar nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if indice is not None:
                arquivo = self.checkpoint_dir / f"{nome}_chunk_{indice}_{timestamp}.csv"
            else:
                arquivo = self.checkpoint_dir / f"{nome}_{timestamp}.csv"

            # Converter listas para strings antes de salvar
            dados_para_salvar = dados.copy()
            if 'historico' in dados_para_salvar.columns:
                dados_para_salvar['historico'] = dados_para_salvar['historico'].apply(json.dumps)
            if 'historicoTimestamp' in dados_para_salvar.columns:
                dados_para_salvar['historicoTimestamp'] = dados_para_salvar['historicoTimestamp'].apply(json.dumps)

            # Salvar dados
            dados_para_salvar.to_csv(arquivo, index=False)
            logger.info(f"Checkpoint salvo: {arquivo}")
            return str(arquivo)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            raise

    def _carregar_checkpoints(self, padrao: str) -> pd.DataFrame:
        """
        Carrega checkpoints salvos.
        
        Args:
            padrao: Padrão do nome dos arquivos a serem carregados
            
        Returns:
            DataFrame combinado dos checkpoints
        """
        try:
            arquivos = list(self.checkpoint_dir.glob(padrao))
            if not arquivos:
                return None

            dfs = []
            for arquivo in arquivos:
                logger.info(f"Carregando checkpoint: {arquivo}")
                df = pd.read_csv(arquivo)
                
                # Converter strings de volta para listas
                if 'historico' in df.columns:
                    df['historico'] = df['historico'].apply(json.loads)
                if 'historicoTimestamp' in df.columns:
                    df['historicoTimestamp'] = df['historicoTimestamp'].apply(json.loads)
                
                dfs.append(df)

            return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoints: {str(e)}")
            raise

    def _limpar_checkpoints(self, padrao: str = None):
        """Limpa arquivos de checkpoint."""
        try:
            if padrao:
                arquivos = list(self.checkpoint_dir.glob(padrao))
            else:
                arquivos = list(self.checkpoint_dir.glob('*'))

            for arquivo in arquivos:
                arquivo.unlink()
            logger.info("Checkpoints removidos")

        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")

    def processar_dados_treino(self, arquivos_treino: List[str], 
                             arquivos_itens: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa os dados de treino e itens usando chunks e salvamento intermediário.
        """
        logger.info("Iniciando processamento dos dados")
        
        # Verificar arquivos
        self._verificar_arquivos(arquivos_treino)
        self._verificar_arquivos(arquivos_itens)
        
        # Verificar estrutura dos arquivos
        if arquivos_itens:
            df_exemplo = pd.read_csv(arquivos_itens[0], nrows=1)
            logger.info(f"Colunas disponíveis no arquivo de itens: {df_exemplo.columns.tolist()}")
        
        try:
            # Limpar checkpoints antigos
            self._limpar_checkpoints()
            
            # Processar dados de treino em chunks
            logger.info("Processando dados de treino")
            checkpoint_files_treino = []
            
            for arquivo in arquivos_treino:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de treino: {arquivo}")
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de treino")
                        logger.info(f"Colunas disponíveis no chunk de treino: {chunk.columns.tolist()}")
                        
                        # Processar chunk
                        chunk = self._verificar_e_renomear_colunas(
                            chunk,
                            {
                                'history': 'historico',
                                'timestampHistory': 'historicoTimestamp',
                                'userId': 'idUsuario'
                            }
                        )
                        
                        chunk['historico'] = chunk['historico'].apply(self._processar_historico)
                        chunk['historicoTimestamp'] = chunk['historicoTimestamp'].apply(
                            self._processar_historico
                        )
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'treino', i)
                        checkpoint_files_treino.append(arquivo_checkpoint)
                        
                        gc.collect()

            # Carregar e combinar checkpoints de treino
            logger.info("Combinando dados de treino")
            dados_treino = self._carregar_checkpoints('treino_chunk_*.csv')
            if dados_treino is None:
                raise ValueError("Falha ao processar dados de treino")

            # Processar dados dos itens
            logger.info("Processando dados de itens")
            checkpoint_files_itens = []
            
            for arquivo in arquivos_itens:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de itens: {arquivo}")
                    # Primeiro, vamos ler apenas uma linha para verificar as colunas
                    df_exemplo = pd.read_csv(arquivo, nrows=1)
                    logger.info(f"Colunas encontradas: {df_exemplo.columns.tolist()}")

                    # Determinar mapeamento baseado nas colunas existentes
                    mapeamento_colunas = self._determinar_mapeamento_colunas(df_exemplo.columns)
                    
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de itens")
                        logger.info(f"Colunas disponíveis no chunk de itens: {chunk.columns.tolist()}")
                        
                        # Usar o mapeamento determinado
                        chunk = self._verificar_e_renomear_colunas(chunk, mapeamento_colunas)
                        
                        # Converter timestamp
                        if 'DataPublicacao' in chunk.columns:
                            chunk['DataPublicacao'] = pd.to_datetime(chunk['DataPublicacao'])
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'itens', i)
                        checkpoint_files_itens.append(arquivo_checkpoint)
                        
                        gc.collect()

            # Carregar e combinar checkpoints de itens
            logger.info("Combinando dados de itens")
            dados_itens = self._carregar_checkpoints('itens_chunk_*.csv')
            if dados_itens is None:
                raise ValueError("Falha ao processar dados de itens")

            # Validar dados finais
            self._validar_colunas_treino(dados_treino)
            self._validar_colunas_itens(dados_itens)

            # Salvar resultado final
            logger.info("Salvando dados processados finais")
            self._salvar_checkpoint(dados_treino, 'treino_final')
            self._salvar_checkpoint(dados_itens, 'itens_final')

            # Limpar checkpoints intermediários
            self._limpar_checkpoints('*chunk*.csv')

            logger.info("Processamento dos dados concluído com sucesso")
            return dados_treino, dados_itens

        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        finally:
            # Garantir que a memória seja liberada
            gc.collect()



    def _validar_colunas_treino(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de treino."""
        logger.info(f"Validando colunas de treino. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['historico', 'historicoTimestamp', 'idUsuario']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de treino: {colunas_faltantes}"
            logger.error(erro_msg)
            raise ValueError(erro_msg)

    def _validar_colunas_itens(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de itens."""
        logger.info(f"Validando colunas dos itens. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['Pagina', 'Titulo', 'Corpo', 'DataPublicacao']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de itens: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)

    def _processar_historico(self, valor: str) -> list:
        """
        Processa strings de histórico para listas.
        
        Args:
            valor: String contendo a lista a ser processada
            
        Returns:
            list: Lista processada
        """
        try:
            if isinstance(valor, list):
                return valor
            if isinstance(valor, str):
                try:
                    return ast.literal_eval(valor)
                except:
                    # Se falhar, tentar limpar a string e processar
                    valor_limpo = valor.strip('[]').replace(' ', '').split(',')
                    return [v.strip("'") for v in valor_limpo if v]
            return []
        except Exception as e:
            logger.error(f"Erro ao processar histórico: {str(e)}")
            logger.error(f"Valor problemático: {valor}")
            return []

    def preparar_features_texto(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """Prepara features de texto combinando título e corpo."""
        logger.info("Preparando features de texto")
        try:
            dados_itens['conteudo_texto'] = dados_itens['Titulo'].fillna('') + ' ' + \
                                          dados_itens['Corpo'].fillna('')
            return dados_itens
        except Exception as e:
            logger.error(f"Erro ao preparar features de texto: {str(e)}")
            raise

    def validar_dados(self, dados_treino: pd.DataFrame, 
                     dados_itens: pd.DataFrame) -> bool:
        """
        Realiza validações adicionais nos dados.
        
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Iniciando validação dos dados")
        try:
            # Verificar valores nulos
            nulos_treino = dados_treino.isnull().sum()
            nulos_itens = dados_itens.isnull().sum()
            
            if nulos_treino.any():
                logger.warning("Valores nulos encontrados nos dados de treino:")
                for coluna, quantidade in nulos_treino[nulos_treino > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")
            
            if nulos_itens.any():
                logger.warning("Valores nulos encontrados nos dados de itens:")
                for coluna, quantidade in nulos_itens[nulos_itens > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")

            # Verificar consistência dos dados
            self._verificar_consistencia_dados(dados_treino, dados_itens)
            
            logger.info("Validação dos dados concluída")
            return True

        except Exception as e:
            logger.error(f"Erro na validação dos dados: {str(e)}")
            return False

    def _verificar_consistencia_dados(self, dados_treino: pd.DataFrame, 
                                    dados_itens: pd.DataFrame) -> None:
        """Verifica a consistência entre dados de treino e itens."""
        try:
            # Verificar se todos os itens do histórico existem nos dados de itens
            todos_itens = set(dados_itens['Pagina'])
            itens_historico = set([
                item for historico in dados_treino['historico'] 
                for item in historico
            ])
            
            itens_faltantes = itens_historico - todos_itens
            if itens_faltantes:
                logger.warning(
                    f"Existem {len(itens_faltantes)} itens no histórico que não existem nos dados de itens"
                )
                
        except Exception as e:
            logger.error(f"Erro ao verificar consistência dos dados: {str(e)}")
            raise

    def mostrar_info_dados(self, dados_treino: pd.DataFrame, 
                          dados_itens: pd.DataFrame) -> None:
        """Mostra informações detalhadas sobre os dados."""
        logger.info("Exibindo informações dos dados")
        try:
            print("\nInformações dos dados de treino:")
            print(f"Número de registros: {len(dados_treino)}")
            print(f"Número de usuários únicos: {dados_treino['idUsuario'].nunique()}")
            print(f"Média de itens por usuário: {dados_treino['historico'].apply(len).mean():.2f}")
            print("\nAmostra dos dados de treino:")
            print(dados_treino.head())
            
            print("\nInformações dos dados de itens:")
            print(f"Número de itens: {len(dados_itens)}")
            print(f"Período dos dados: {dados_itens['DataPublicacao'].min()} até {dados_itens['DataPublicacao'].max()}")
            print("\nAmostra dos dados de itens:")
            print(dados_itens.head())
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações dos dados: {str(e)}")
            raise'''

'''import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Tuple, List
import ast
import traceback
import gc
import json
from pathlib import Path
import shutil
import psutil

logger = logging.getLogger(__name__)

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None
        self.chunk_size = 10000
        self.chunk_size_texto = 1000  # Tamanho específico para processamento de texto
        self.checkpoint_dir = Path('dados/processados/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Inicializando PreProcessadorDados")

    def _monitorar_memoria(self):
        """Monitora o uso de memória atual."""
        processo = psutil.Process(os.getpid())
        uso_memoria = processo.memory_info().rss / 1024 / 1024  # Converter para MB
        logger.info(f"Uso atual de memória: {uso_memoria:.2f} MB")

    def _verificar_arquivos(self, arquivos: List[str]) -> None:
        """Verifica o tamanho e existência dos arquivos."""
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo) / (1024 * 1024)  # MB
                logger.info(f"Arquivo {arquivo}: {tamanho:.2f} MB")
            else:
                logger.warning(f"Arquivo não encontrado: {arquivo}")

    def _determinar_mapeamento_colunas(self, colunas_disponiveis: List[str]) -> dict:
        """
        Determina o mapeamento correto de colunas baseado nas colunas disponíveis.
        """
        # Possíveis nomes para cada coluna
        mapeamentos_possiveis = {
            'Pagina': ['Page', 'page', 'PAGE', 'url', 'URL', 'link'],
            'Titulo': ['Title', 'title', 'TITLE', 'titulo', 'Título'],
            'Corpo': ['Body', 'body', 'BODY', 'texto', 'content', 'Content'],
            'DataPublicacao': ['Issued', 'issued', 'ISSUED', 'data', 'date', 'Data', 'published_date']
        }

        mapeamento_final = {}
        colunas_disponiveis = set(colunas_disponiveis)
        
        for coluna_final, possiveis_nomes in mapeamentos_possiveis.items():
            for nome in possiveis_nomes:
                if nome in colunas_disponiveis:
                    mapeamento_final[nome] = coluna_final
                    break
        
        logger.info(f"Mapeamento de colunas determinado: {mapeamento_final}")
        return mapeamento_final

    def _verificar_e_renomear_colunas(self, df: pd.DataFrame, mapeamento: dict) -> pd.DataFrame:
        """
        Verifica se as colunas existem e renomeia.
        
        Args:
            df: DataFrame a ser processado
            mapeamento: Dicionário com o mapeamento de colunas
            
        Returns:
            DataFrame com colunas renomeadas
        """
        logger.info(f"Verificando e renomeando colunas. Colunas disponíveis: {df.columns.tolist()}")
        logger.info(f"Mapeamento a ser aplicado: {mapeamento}")
        
        # Verificar se as colunas originais existem
        colunas_faltantes = [col for col in mapeamento.keys() if col not in df.columns]
        if colunas_faltantes:
            erro_msg = f"Colunas originais faltantes: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)
        
        return df.rename(columns=mapeamento)



    def _salvar_checkpoint(self, dados: pd.DataFrame, nome: str, indice: int = None) -> str:
        """
        Salva um checkpoint dos dados processados.
        
        Args:
            dados: DataFrame a ser salvo
            nome: Nome base do arquivo
            indice: Índice do chunk (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        try:
            # Criar nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if indice is not None:
                arquivo = self.checkpoint_dir / f"{nome}_chunk_{indice}_{timestamp}.csv"
            else:
                arquivo = self.checkpoint_dir / f"{nome}_{timestamp}.csv"

            # Converter listas para strings antes de salvar
            dados_para_salvar = dados.copy()
            if 'historico' in dados_para_salvar.columns:
                dados_para_salvar['historico'] = dados_para_salvar['historico'].apply(json.dumps)
            if 'historicoTimestamp' in dados_para_salvar.columns:
                dados_para_salvar['historicoTimestamp'] = dados_para_salvar['historicoTimestamp'].apply(json.dumps)

            # Salvar dados
            dados_para_salvar.to_csv(arquivo, index=False)
            logger.info(f"Checkpoint salvo: {arquivo}")
            return str(arquivo)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            raise

    def _carregar_checkpoints(self, padrao: str) -> pd.DataFrame:
        """
        Carrega checkpoints salvos.
        
        Args:
            padrao: Padrão do nome dos arquivos a serem carregados
            
        Returns:
            DataFrame combinado dos checkpoints
        """
        try:
            arquivos = list(self.checkpoint_dir.glob(padrao))
            if not arquivos:
                return None

            dfs = []
            for arquivo in arquivos:
                logger.info(f"Carregando checkpoint: {arquivo}")
                df = pd.read_csv(arquivo)
                
                # Converter strings de volta para listas
                if 'historico' in df.columns:
                    df['historico'] = df['historico'].apply(json.loads)
                if 'historicoTimestamp' in df.columns:
                    df['historicoTimestamp'] = df['historicoTimestamp'].apply(json.loads)
                
                dfs.append(df)

            return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoints: {str(e)}")
            raise

    def _limpar_checkpoints(self, padrao: str = None):
        """Limpa arquivos de checkpoint."""
        try:
            if padrao:
                arquivos = list(self.checkpoint_dir.glob(padrao))
            else:
                arquivos = list(self.checkpoint_dir.glob('*'))

            for arquivo in arquivos:
                arquivo.unlink()
            logger.info("Checkpoints removidos")

        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")

    def _processar_texto_em_chunks(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """
        Processa as features de texto em chunks para economizar memória.
        """
        logger.info("Iniciando processamento de texto em chunks")
        self._monitorar_memoria()

        total_rows = len(dados_itens)
        
        for i in range(0, total_rows, self.chunk_size_texto):
            chunk_end = min(i + self.chunk_size_texto, total_rows)
            logger.info(f"Processando chunk de texto {i//self.chunk_size_texto + 1} "
                       f"de {total_rows//self.chunk_size_texto + 1}")
            
            # Processar chunk atual
            chunk = dados_itens.iloc[i:chunk_end].copy()
            chunk['conteudo_texto'] = chunk['Titulo'].fillna('') + ' ' + chunk['Corpo'].fillna('')
            dados_itens.loc[i:chunk_end-1, 'conteudo_texto'] = chunk['conteudo_texto']
            
            # Limpar memória
            del chunk
            gc.collect()
            
            # Monitorar memória periodicamente
            if (i//self.chunk_size_texto) % 10 == 0:
                self._monitorar_memoria()

        logger.info("Processamento de texto em chunks concluído")
        return dados_itens


    def processar_dados_treino(self, arquivos_treino: List[str], 
                             arquivos_itens: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa os dados de treino e itens usando chunks e salvamento intermediário.
        """
        logger.info("Iniciando processamento dos dados")
        self._monitorar_memoria()
        
        # Verificar arquivos
        self._verificar_arquivos(arquivos_treino)
        self._verificar_arquivos(arquivos_itens)
        
        # Verificar estrutura dos arquivos
        if arquivos_itens:
            df_exemplo = pd.read_csv(arquivos_itens[0], nrows=1)
            logger.info(f"Colunas disponíveis no arquivo de itens: {df_exemplo.columns.tolist()}")
        
        try:
            # Limpar checkpoints antigos
            self._limpar_checkpoints()
            
            # Processar dados de treino em chunks
            logger.info("Processando dados de treino")
            checkpoint_files_treino = []
            
            for arquivo in arquivos_treino:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de treino: {arquivo}")
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de treino")
                        logger.info(f"Colunas disponíveis no chunk de treino: {chunk.columns.tolist()}")
                        
                        # Processar chunk
                        chunk = self._verificar_e_renomear_colunas(
                            chunk,
                            {
                                'history': 'historico',
                                'timestampHistory': 'historicoTimestamp',
                                'userId': 'idUsuario'
                            }
                        )
                        
                        chunk['historico'] = chunk['historico'].apply(self._processar_historico)
                        chunk['historicoTimestamp'] = chunk['historicoTimestamp'].apply(
                            self._processar_historico
                        )
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'treino', i)
                        checkpoint_files_treino.append(arquivo_checkpoint)
                        
                        # Monitorar e limpar memória
                        if i % 10 == 0:
                            self._monitorar_memoria()
                        gc.collect()

            # Carregar e combinar checkpoints de treino
            logger.info("Combinando dados de treino")
            dados_treino = self._carregar_checkpoints('treino_chunk_*.csv')
            if dados_treino is None:
                raise ValueError("Falha ao processar dados de treino")

            # Processar dados dos itens
            logger.info("Processando dados de itens")
            checkpoint_files_itens = []
            
            for arquivo in arquivos_itens:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de itens: {arquivo}")
                    # Primeiro, vamos ler apenas uma linha para verificar as colunas
                    df_exemplo = pd.read_csv(arquivo, nrows=1)
                    logger.info(f"Colunas encontradas: {df_exemplo.columns.tolist()}")

                    # Determinar mapeamento baseado nas colunas existentes
                    mapeamento_colunas = self._determinar_mapeamento_colunas(df_exemplo.columns)
                    
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processando chunk {i+1} de itens")
                        logger.info(f"Colunas disponíveis no chunk de itens: {chunk.columns.tolist()}")
                        
                        # Usar o mapeamento determinado
                        chunk = self._verificar_e_renomear_colunas(chunk, mapeamento_colunas)
                        
                        # Converter timestamp
                        if 'DataPublicacao' in chunk.columns:
                            chunk['DataPublicacao'] = pd.to_datetime(chunk['DataPublicacao'])
                        
                        # Salvar checkpoint
                        arquivo_checkpoint = self._salvar_checkpoint(chunk, 'itens', i)
                        checkpoint_files_itens.append(arquivo_checkpoint)
                        
                        # Monitorar e limpar memória
                        if i % 10 == 0:
                            self._monitorar_memoria()
                        gc.collect()

            # Carregar e combinar checkpoints de itens
            logger.info("Combinando dados de itens")
            dados_itens = self._carregar_checkpoints('itens_chunk_*.csv')
            if dados_itens is None:
                raise ValueError("Falha ao processar dados de itens")

            # Validar dados finais
            self._validar_colunas_treino(dados_treino)
            self._validar_colunas_itens(dados_itens)

            # Processar features de texto em chunks
            logger.info("Iniciando processamento de features de texto")
            dados_itens = self._processar_texto_em_chunks(dados_itens)

            # Salvar resultado final
            logger.info("Salvando dados processados finais")
            self._salvar_checkpoint(dados_treino, 'treino_final')
            self._salvar_checkpoint(dados_itens, 'itens_final')

            # Limpar checkpoints intermediários
            self._limpar_checkpoints('*chunk*.csv')

            logger.info("Processamento dos dados concluído com sucesso")
            return dados_treino, dados_itens

        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        finally:
            gc.collect()
            self._monitorar_memoria()


    def _validar_colunas_treino(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de treino."""
        logger.info(f"Validando colunas de treino. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['historico', 'historicoTimestamp', 'idUsuario']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de treino: {colunas_faltantes}"
            logger.error(erro_msg)
            raise ValueError(erro_msg)

    def _validar_colunas_itens(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de itens."""
        logger.info(f"Validando colunas dos itens. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['Pagina', 'Titulo', 'Corpo', 'DataPublicacao']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de itens: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)

    def _processar_historico(self, valor: str) -> list:
        """
        Processa strings de histórico para listas.
        
        Args:
            valor: String contendo a lista a ser processada
            
        Returns:
            list: Lista processada
        """
        try:
            if isinstance(valor, list):
                return valor
            if isinstance(valor, str):
                try:
                    return ast.literal_eval(valor)
                except:
                    # Se falhar, tentar limpar a string e processar
                    valor_limpo = valor.strip('[]').replace(' ', '').split(',')
                    return [v.strip("'") for v in valor_limpo if v]
            return []
        except Exception as e:
            logger.error(f"Erro ao processar histórico: {str(e)}")
            logger.error(f"Valor problemático: {valor}")
            return []

    def preparar_features_texto(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """Prepara features de texto combinando título e corpo."""
        logger.info("Preparando features de texto")
        try:
            # Usar o método de processamento em chunks
            return self._processar_texto_em_chunks(dados_itens)
        except Exception as e:
            logger.error(f"Erro ao preparar features de texto: {str(e)}")
            raise

    def validar_dados(self, dados_treino: pd.DataFrame, 
                     dados_itens: pd.DataFrame) -> bool:
        """
        Realiza validações adicionais nos dados.
        
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Iniciando validação dos dados")
        self._monitorar_memoria()
        
        try:
            # Verificar valores nulos
            nulos_treino = dados_treino.isnull().sum()
            nulos_itens = dados_itens.isnull().sum()
            
            if nulos_treino.any():
                logger.warning("Valores nulos encontrados nos dados de treino:")
                for coluna, quantidade in nulos_treino[nulos_treino > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")
            
            if nulos_itens.any():
                logger.warning("Valores nulos encontrados nos dados de itens:")
                for coluna, quantidade in nulos_itens[nulos_itens > 0].items():
                    logger.warning(f"{coluna}: {quantidade} valores nulos")

            # Verificar consistência dos dados
            self._verificar_consistencia_dados(dados_treino, dados_itens)
            
            logger.info("Validação dos dados concluída")
            return True

        except Exception as e:
            logger.error(f"Erro na validação dos dados: {str(e)}")
            return False
        finally:
            self._monitorar_memoria()

    def _verificar_consistencia_dados(self, dados_treino: pd.DataFrame, 
                                    dados_itens: pd.DataFrame) -> None:
        """Verifica a consistência entre dados de treino e itens."""
        try:
            # Verificar se todos os itens do histórico existem nos dados de itens
            todos_itens = set(dados_itens['Pagina'])
            
            # Processar histórico em chunks para economizar memória
            itens_historico = set()
            chunk_size = 1000
            
            for i in range(0, len(dados_treino), chunk_size):
                chunk = dados_treino.iloc[i:i+chunk_size]
                novos_itens = set([
                    item for historico in chunk['historico'] 
                    for item in historico
                ])
                itens_historico.update(novos_itens)
                
                if i % (chunk_size * 10) == 0:
                    self._monitorar_memoria()
            
            itens_faltantes = itens_historico - todos_itens
            if itens_faltantes:
                logger.warning(
                    f"Existem {len(itens_faltantes)} itens no histórico que não existem nos dados de itens"
                )
                
        except Exception as e:
            logger.error(f"Erro ao verificar consistência dos dados: {str(e)}")
            raise

    def mostrar_info_dados(self, dados_treino: pd.DataFrame, 
                          dados_itens: pd.DataFrame) -> None:
        """Mostra informações detalhadas sobre os dados."""
        logger.info("Exibindo informações dos dados")
        self._monitorar_memoria()
        
        try:
            print("\nInformações dos dados de treino:")
            print(f"Número de registros: {len(dados_treino)}")
            print(f"Número de usuários únicos: {dados_treino['idUsuario'].nunique()}")
            print(f"Média de itens por usuário: {dados_treino['historico'].apply(len).mean():.2f}")
            print("\nAmostra dos dados de treino:")
            print(dados_treino.head())
            
            print("\nInformações dos dados de itens:")
            print(f"Número de itens: {len(dados_itens)}")
            print(f"Período dos dados: {dados_itens['DataPublicacao'].min()} até {dados_itens['DataPublicacao'].max()}")
            print("\nAmostra dos dados de itens:")
            print(dados_itens.head())
            
        except Exception as e:
            logger.error(f"Erro ao mostrar informações dos dados: {str(e)}")
            raise
        finally:
            self._monitorar_memoria()'''



'''import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Tuple, List
import ast
import traceback
import gc
import json
from pathlib import Path
import shutil
import psutil

logger = logging.getLogger(__name__)

class PreProcessadorDados:
    def __init__(self):
        self.dados_processados = None
        self.chunk_size = 5000  # Reduzido para processamento mais leve
        self.chunk_size_texto = 50  # Reduzido para processamento mais leve
        self.checkpoint_dir = Path('dados/processados/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar limites de memória mais conservadores
        self.limite_memoria_mb = 1024  # 1GB
        self.grupo_size_checkpoints = 3  # Número de checkpoints a processar por vez
        logger.info("Inicializando PreProcessadorDados")

    def _verificar_limite_memoria(self):
        """Verifica se o uso de memória está próximo do limite."""
        processo = psutil.Process(os.getpid())
        uso_memoria = processo.memory_info().rss / 1024 / 1024  # MB
        
        if uso_memoria > self.limite_memoria_mb:
            logger.warning(f"Uso de memória ({uso_memoria:.2f}MB) excedeu limite ({self.limite_memoria_mb}MB)")
            logger.info("Forçando coleta de lixo")
            gc.collect()
            
            # Verificar novamente após gc
            uso_memoria = processo.memory_info().rss / 1024 / 1024
            if uso_memoria > self.limite_memoria_mb:
                raise MemoryError(f"Uso de memória ({uso_memoria:.2f}MB) ainda excede limite após gc")

    def _monitorar_memoria(self):
        """Monitora o uso de memória atual."""
        processo = psutil.Process(os.getpid())
        uso_memoria = processo.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Uso atual de memória: {uso_memoria:.2f} MB")

    def _verificar_arquivos(self, arquivos: List[str]) -> None:
        """Verifica o tamanho e existência dos arquivos."""
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo) / (1024 * 1024)  # MB
                logger.info(f"Arquivo {arquivo}: {tamanho:.2f} MB")
            else:
                logger.warning(f"Arquivo não encontrado: {arquivo}")


    def _determinar_mapeamento_colunas(self, colunas_disponiveis: List[str]) -> dict:
        """
        Determina o mapeamento correto de colunas baseado nas colunas disponíveis.
        """
        # Possíveis nomes para cada coluna
        mapeamentos_possiveis = {
            'Pagina': ['Page', 'page', 'PAGE', 'url', 'URL', 'link'],
            'Titulo': ['Title', 'title', 'TITLE', 'titulo', 'Título'],
            'Corpo': ['Body', 'body', 'BODY', 'texto', 'content', 'Content'],
            'DataPublicacao': ['Issued', 'issued', 'ISSUED', 'data', 'date', 'Data', 'published_date']
        }

        mapeamento_final = {}
        colunas_disponiveis = set(colunas_disponiveis)
        
        for coluna_final, possiveis_nomes in mapeamentos_possiveis.items():
            for nome in possiveis_nomes:
                if nome in colunas_disponiveis:
                    mapeamento_final[nome] = coluna_final
                    break
        
        logger.info(f"Mapeamento de colunas determinado: {mapeamento_final}")
        return mapeamento_final

    def _verificar_e_renomear_colunas(self, df: pd.DataFrame, mapeamento: dict) -> pd.DataFrame:
        """
        Verifica se as colunas existem e renomeia.
        
        Args:
            df: DataFrame a ser processado
            mapeamento: Dicionário com o mapeamento de colunas
            
        Returns:
            DataFrame com colunas renomeadas
        """
        logger.info(f"Verificando e renomeando colunas. Colunas disponíveis: {df.columns.tolist()}")
        logger.info(f"Mapeamento a ser aplicado: {mapeamento}")
        
        # Verificar se as colunas originais existem
        colunas_faltantes = [col for col in mapeamento.keys() if col not in df.columns]
        if colunas_faltantes:
            erro_msg = f"Colunas originais faltantes: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)
        
        return df.rename(columns=mapeamento)

    def _salvar_checkpoint(self, dados: pd.DataFrame, nome: str, indice: int = None) -> str:
        """
        Salva um checkpoint dos dados processados.
        
        Args:
            dados: DataFrame a ser salvo
            nome: Nome base do arquivo
            indice: Índice do chunk (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        try:
            # Criar nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if indice is not None:
                arquivo = self.checkpoint_dir / f"{nome}_chunk_{indice}_{timestamp}.csv"
            else:
                arquivo = self.checkpoint_dir / f"{nome}_{timestamp}.csv"

            # Converter listas para strings antes de salvar
            dados_para_salvar = dados.copy()
            if 'historico' in dados_para_salvar.columns:
                dados_para_salvar['historico'] = dados_para_salvar['historico'].apply(json.dumps)
            if 'historicoTimestamp' in dados_para_salvar.columns:
                dados_para_salvar['historicoTimestamp'] = dados_para_salvar['historicoTimestamp'].apply(json.dumps)

            # Salvar dados
            dados_para_salvar.to_csv(arquivo, index=False)
            logger.info(f"Checkpoint salvo: {arquivo}")
            
            # Verificar memória após salvar
            self._verificar_limite_memoria()
            
            return str(arquivo)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            raise

    def _carregar_checkpoints(self, padrao: str) -> pd.DataFrame:
        """
        Carrega checkpoints salvos em grupos pequenos para economizar memória.
        
        Args:
            padrao: Padrão do nome dos arquivos a serem carregados
            
        Returns:
            DataFrame combinado dos checkpoints
        """
        try:
            arquivos = sorted(list(self.checkpoint_dir.glob(padrao)))
            if not arquivos:
                return None

            logger.info(f"Total de checkpoints a carregar: {len(arquivos)}")
            resultado_final = None
            
            # Processar arquivos em grupos pequenos
            for i in range(0, len(arquivos), self.grupo_size_checkpoints):
                grupo_atual = arquivos[i:i+self.grupo_size_checkpoints]
                logger.info(f"Processando grupo {i//self.grupo_size_checkpoints + 1} "
                          f"de {len(arquivos)//self.grupo_size_checkpoints + 1}")
                
                dfs_grupo = []
                for arquivo in grupo_atual:
                    try:
                        logger.info(f"Carregando checkpoint: {arquivo}")
                        df = pd.read_csv(arquivo)
                        
                        # Converter strings de volta para listas
                        if 'historico' in df.columns:
                            df['historico'] = df['historico'].apply(json.loads)
                        if 'historicoTimestamp' in df.columns:
                            df['historicoTimestamp'] = df['historicoTimestamp'].apply(json.loads)
                        
                        dfs_grupo.append(df)
                        
                        # Limpar arquivo após processamento
                        os.remove(arquivo)
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar arquivo {arquivo}: {str(e)}")
                        raise
                
                # Combinar DataFrames do grupo atual
                df_grupo = pd.concat(dfs_grupo, ignore_index=True)
                del dfs_grupo
                gc.collect()
                
                # Adicionar ao resultado final
                if resultado_final is None:
                    resultado_final = df_grupo
                else:
                    resultado_final = pd.concat([resultado_final, df_grupo], ignore_index=True)
                    del df_grupo
                    gc.collect()
                
                # Verificar memória após cada grupo
                self._monitorar_memoria()
                if i % (self.grupo_size_checkpoints * 2) == 0:
                    gc.collect()

            return resultado_final

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoints: {str(e)}")
            raise

    def _limpar_checkpoints(self, padrao: str = None):
        """Limpa arquivos de checkpoint."""
        try:
            if padrao:
                arquivos = list(self.checkpoint_dir.glob(padrao))
            else:
                arquivos = list(self.checkpoint_dir.glob('*'))

            for arquivo in arquivos:
                try:
                    arquivo.unlink()
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo {arquivo}: {str(e)}")
                    
            logger.info("Checkpoints removidos")

        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {str(e)}")

    def _processar_texto_em_chunks(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """
        Processa as features de texto em chunks muito menores e com salvamento intermediário.
        """
        logger.info("Iniciando processamento de texto em chunks")
        self._monitorar_memoria()

        # Criar DataFrame temporário para resultados
        temp_dir = self.checkpoint_dir / 'temp_texto'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            total_rows = len(dados_itens)
            checkpoints_texto = []
            
            for i in range(0, total_rows, self.chunk_size_texto):
                chunk_end = min(i + self.chunk_size_texto, total_rows)
                chunk_num = i//self.chunk_size_texto + 1
                total_chunks = (total_rows//self.chunk_size_texto) + 1
                
                logger.info(f"Processando chunk de texto {chunk_num} de {total_chunks}")
                self._monitorar_memoria()
                
                try:
                    # Processar chunk atual com cópia mínima de dados
                    indices = dados_itens.index[i:chunk_end]
                    
                    # Processar texto em séries separadas para economizar memória
                    titulos = dados_itens.loc[indices, 'Titulo'].fillna('')
                    corpos = dados_itens.loc[indices, 'Corpo'].fillna('')
                    
                    # Combinar textos de forma eficiente
                    conteudo_texto = titulos + ' ' + corpos
                    
                    # Liberar memória imediatamente
                    del titulos
                    del corpos
                    gc.collect()


                    # Criar DataFrame temporário apenas com o necessário
                    temp_df = pd.DataFrame({
                        'index_original': indices,
                        'conteudo_texto': conteudo_texto
                    })
                    
                    # Salvar checkpoint
                    checkpoint_path = temp_dir / f'texto_chunk_{chunk_num}.parquet'
                    temp_df.to_parquet(checkpoint_path)
                    checkpoints_texto.append(checkpoint_path)
                    
                    # Liberar memória
                    del temp_df
                    del conteudo_texto
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Erro no chunk {chunk_num}: {str(e)}")
                    raise
                
                # Verificar limite de memória
                self._verificar_limite_memoria()
                
                # Forçar coleta de lixo a cada 10 chunks
                if chunk_num % 10 == 0:
                    gc.collect()
                    self._monitorar_memoria()
            
            # Combinar resultados
            logger.info("Combinando resultados do processamento de texto")
            
            # Processar checkpoints em grupos pequenos
            grupo_size = 2  # Reduzido para menor uso de memória
            for i in range(0, len(checkpoints_texto), grupo_size):
                grupo_atual = checkpoints_texto[i:i+grupo_size]
                
                # Carregar e processar grupo de checkpoints
                for checkpoint in grupo_atual:
                    try:
                        temp_df = pd.read_parquet(checkpoint)
                        indices = temp_df['index_original'].values
                        dados_itens.loc[indices, 'conteudo_texto'] = temp_df['conteudo_texto'].values
                        
                        # Limpar arquivo temporário
                        os.remove(checkpoint)
                        
                        gc.collect()
                        self._verificar_limite_memoria()
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar checkpoint {checkpoint}: {str(e)}")
                        raise
            
            logger.info("Processamento de texto em chunks concluído")
            return dados_itens
            
        except Exception as e:
            logger.error(f"Erro no processamento de texto: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        finally:
            # Limpar diretório temporário
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Erro ao limpar diretório temporário: {str(e)}")
            
            gc.collect()
            self._monitorar_memoria()


    def processar_dados_treino(self, arquivos_treino: List[str], 
                             arquivos_itens: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa os dados de treino e itens usando chunks e salvamento intermediário.
        """
        logger.info("Iniciando processamento dos dados")
        self._monitorar_memoria()
        
        # Verificar arquivos
        self._verificar_arquivos(arquivos_treino)
        self._verificar_arquivos(arquivos_itens)
        
        # Verificar estrutura dos arquivos
        if arquivos_itens:
            df_exemplo = pd.read_csv(arquivos_itens[0], nrows=1)
            logger.info(f"Colunas disponíveis no arquivo de itens: {df_exemplo.columns.tolist()}")
        
        try:
            # Limpar checkpoints antigos
            self._limpar_checkpoints()
            
            # Processar dados de treino em chunks
            logger.info("Processando dados de treino")
            checkpoint_files_treino = []
            
            for arquivo in arquivos_treino:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de treino: {arquivo}")
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            logger.debug(f"Processando chunk {i+1} de treino")
                            logger.info(f"Colunas disponíveis no chunk de treino: {chunk.columns.tolist()}")
                            
                            # Processar chunk
                            chunk = self._verificar_e_renomear_colunas(
                                chunk,
                                {
                                    'history': 'historico',
                                    'timestampHistory': 'historicoTimestamp',
                                    'userId': 'idUsuario'
                                }
                            )
                            
                            chunk['historico'] = chunk['historico'].apply(self._processar_historico)
                            chunk['historicoTimestamp'] = chunk['historicoTimestamp'].apply(
                                self._processar_historico
                            )
                            
                            # Salvar checkpoint
                            arquivo_checkpoint = self._salvar_checkpoint(chunk, 'treino', i)
                            checkpoint_files_treino.append(arquivo_checkpoint)
                            
                            # Verificar memória
                            self._verificar_limite_memoria()
                            
                        except Exception as e:
                            logger.error(f"Erro processando chunk de treino {i+1}: {str(e)}")
                            raise
                        
                        # Limpar memória periodicamente
                        if i % 5 == 0:
                            gc.collect()
                            self._monitorar_memoria()
                else:
                    logger.warning(f"Arquivo não encontrado: {arquivo}")

            # Carregar e combinar checkpoints de treino
            logger.info("Combinando dados de treino")
            dados_treino = self._carregar_checkpoints('treino_chunk_*.csv')
            if dados_treino is None:
                raise ValueError("Falha ao processar dados de treino")

            # Processar dados dos itens
            logger.info("Processando dados de itens")
            checkpoint_files_itens = []

            for arquivo in arquivos_itens:
                if os.path.exists(arquivo):
                    logger.info(f"Processando arquivo de itens: {arquivo}")
                    # Primeiro, vamos ler apenas uma linha para verificar as colunas
                    df_exemplo = pd.read_csv(arquivo, nrows=1)
                    logger.info(f"Colunas encontradas: {df_exemplo.columns.tolist()}")

                    # Determinar mapeamento baseado nas colunas existentes
                    mapeamento_colunas = self._determinar_mapeamento_colunas(df_exemplo.columns)
                    
                    chunks = pd.read_csv(arquivo, chunksize=self.chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            logger.debug(f"Processando chunk {i+1} de itens")
                            logger.info(f"Colunas disponíveis no chunk de itens: {chunk.columns.tolist()}")
                            
                            # Usar o mapeamento determinado
                            chunk = self._verificar_e_renomear_colunas(chunk, mapeamento_colunas)
                            
                            # Converter timestamp
                            if 'DataPublicacao' in chunk.columns:
                                chunk['DataPublicacao'] = pd.to_datetime(chunk['DataPublicacao'])
                            
                            # Salvar checkpoint
                            arquivo_checkpoint = self._salvar_checkpoint(chunk, 'itens', i)
                            checkpoint_files_itens.append(arquivo_checkpoint)
                            
                            # Verificar memória
                            self._verificar_limite_memoria()
                            
                        except Exception as e:
                            logger.error(f"Erro processando chunk de itens {i+1}: {str(e)}")
                            raise
                        
                        # Limpar memória periodicamente
                        if i % 5 == 0:
                            gc.collect()
                            self._monitorar_memoria()
                else:
                    logger.warning(f"Arquivo de itens não encontrado: {arquivo}")

            # Carregar e combinar checkpoints de itens
            logger.info("Combinando dados de itens")
            dados_itens = self._carregar_checkpoints('itens_chunk_*.csv')
            if dados_itens is None:
                raise ValueError("Falha ao processar dados de itens")

            # Validar dados finais
            self._validar_colunas_treino(dados_treino)
            self._validar_colunas_itens(dados_itens)

            # Processar features de texto em chunks menores
            logger.info("Iniciando processamento de features de texto")
            self._monitorar_memoria()
            
            try:
                dados_itens = self._processar_texto_em_chunks(dados_itens)
            except MemoryError:
                logger.error("Erro de memória durante processamento de texto")
                raise
            except Exception as e:
                logger.error(f"Erro durante processamento de texto: {str(e)}")
                raise

            # Salvar resultado final
            logger.info("Salvando dados processados finais")
            self._salvar_checkpoint(dados_treino, 'treino_final')
            self._salvar_checkpoint(dados_itens, 'itens_final')

            # Limpar checkpoints intermediários
            self._limpar_checkpoints('*chunk*.csv')

            logger.info("Processamento dos dados concluído com sucesso")
            return dados_treino, dados_itens

        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        finally:
            gc.collect()
            self._monitorar_memoria()


    def _validar_colunas_treino(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de treino."""
        logger.info(f"Validando colunas de treino. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['historico', 'historicoTimestamp', 'idUsuario']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de treino: {colunas_faltantes}"
            logger.error(erro_msg)
            raise ValueError(erro_msg)

    def _validar_colunas_itens(self, df: pd.DataFrame) -> None:
        """Valida as colunas necessárias no DataFrame de itens."""
        logger.info(f"Validando colunas dos itens. Colunas disponíveis: {df.columns.tolist()}")
        colunas_necessarias = ['Pagina', 'Titulo', 'Corpo', 'DataPublicacao']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            erro_msg = f"Colunas faltantes nos dados de itens: {colunas_faltantes}"
            logger.error(erro_msg)
            logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            raise ValueError(erro_msg)

    def _processar_historico(self, valor: str) -> list:
        """
        Processa strings de histórico para listas.
        
        Args:
            valor: String contendo a lista a ser processada
            
        Returns:
            list: Lista processada
        """
        try:
            if isinstance(valor, list):
                return valor
            if isinstance(valor, str):
                try:
                    return ast.literal_eval(valor)
                except:
                    # Se falhar, tentar limpar a string e processar
                    valor_limpo = valor.strip('[]').replace(' ', '').split(',')
                    return [v.strip("'") for v in valor_limpo if v]
            return []
        except Exception as e:
            logger.error(f"Erro ao processar histórico: {str(e)}")
            logger.error(f"Valor problemático: {valor}")
            return []

    def preparar_features_texto(self, dados_itens: pd.DataFrame) -> pd.DataFrame:
        """Prepara features de texto combinando título e corpo."""
        logger.info("Preparando features de texto")
        try:
            return self._processar_texto_em_chunks(dados_itens)
        except Exception as e:
            logger.error(f"Erro ao preparar features de texto: {str(e)}")
            raise'''

    