import unittest
import pandas as pd
import numpy as np
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
import tempfile
import os
import json

class TestPreProcessadorSpark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuração inicial para todos os testes."""
        cls.preprocessador = PreProcessadorDadosSpark(
            memoria_executor="2g",
            memoria_driver="2g"
        )
        
        # Criar dados de teste
        cls.temp_dir = tempfile.mkdtemp()
        cls._criar_dados_teste()

    @classmethod
    def tearDownClass(cls):
        """Limpeza após todos os testes."""
        cls.preprocessador.__del__()
        # Limpar arquivos temporários
        import shutil
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def _criar_dados_teste(cls):
        """Cria arquivos de dados para teste."""
        # Dados de treino
        dados_treino = pd.DataFrame({
            'userId': ['user1', 'user2'],
            'history': [json.dumps(['item1', 'item2']), 
                       json.dumps(['item3', 'item4'])],
            'timestampHistory': [json.dumps([1000, 2000]), 
                               json.dumps([3000, 4000])]
        })
        
        # Dados de itens
        dados_itens = pd.DataFrame({
            'Page': ['item1', 'item2', 'item3', 'item4'],
            'Title': ['Título 1', 'Título 2', 'Título 3', 'Título 4'],
            'Body': ['Corpo 1', 'Corpo 2', 'Corpo 3', 'Corpo 4'],
            'Issued': pd.date_range('2023-01-01', periods=4)
        })
        
        # Salvar arquivos
        cls.arquivo_treino = os.path.join(cls.temp_dir, 'treino_teste.csv')
        cls.arquivo_itens = os.path.join(cls.temp_dir, 'itens_teste.csv')
        
        dados_treino.to_csv(cls.arquivo_treino, index=False)
        dados_itens.to_csv(cls.arquivo_itens, index=False)

    def test_processamento_dados(self):
        """Testa o processamento básico dos dados."""
        dados_treino, dados_itens = self.preprocessador.processar_dados_treino(
            [self.arquivo_treino],
            [self.arquivo_itens]
        )
        
        # Verificar estrutura dos dados
        self.assertIn('historico', dados_treino.columns)
        self.assertIn('historicoTimestamp', dados_treino.columns)
        self.assertIn('idUsuario', dados_treino.columns)
        
        self.assertIn('Pagina', dados_itens.columns)
        self.assertIn('Titulo', dados_itens.columns)
        self.assertIn('Corpo', dados_itens.columns)
        self.assertIn('DataPublicacao', dados_itens.columns)

    def test_validacao_dados(self):
        """Testa a validação dos dados."""
        dados_treino, dados_itens = self.preprocessador.processar_dados_treino(
            [self.arquivo_treino],
            [self.arquivo_itens]
        )
        
        resultado = self.preprocessador.validar_dados(dados_treino, dados_itens)
        self.assertTrue(resultado)

    def test_processamento_texto(self):
        """Testa o processamento de features de texto."""
        _, dados_itens = self.preprocessador.processar_dados_treino(
            [self.arquivo_treino],
            [self.arquivo_itens]
        )
        
        self.assertIn('conteudo_texto', dados_itens.columns)
        self.assertTrue(all(dados_itens['conteudo_texto'].str.contains(' ')))

if __name__ == '__main__':
    unittest.main()