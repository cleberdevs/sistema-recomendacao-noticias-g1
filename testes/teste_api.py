'''import unittest
import json
from src.api.app import app
from src.config.logging_config import configurar_logging
import logging

# Configurar logging para testes
configurar_logging()
logger = logging.getLogger(__name__)

class TesteAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuração inicial para todos os testes."""
        logger.info("Iniciando suite de testes da API")
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def setUp(self):
        """Configuração para cada teste individual."""
        logger.info(f"Iniciando teste: {self._testMethodName}")

    def tearDown(self):
        """Limpeza após cada teste."""
        logger.info(f"Finalizando teste: {self._testMethodName}")

    def test_verificar_saude(self):
        """Testa o endpoint de verificação de saúde."""
        logger.info("Testando endpoint de saúde")
        try:
            resposta = self.client.get('/saude')
            dados = json.loads(resposta.data.decode())
            
            self.assertEqual(resposta.status_code, 200)
            self.assertEqual(dados['status'], 'saudavel')
            self.assertIn('versao_modelo', dados)
            self.assertIn('timestamp', dados)
            
        except Exception as e:
            logger.error(f"Erro no teste de saúde: {str(e)}")
            raise

    def test_endpoint_previsao_sucesso(self):
        """Testa o endpoint de previsão com dados válidos."""
        logger.info("Testando endpoint de previsão - caso de sucesso")
        try:
            entrada_teste = {
                "id_usuario": "usuario_teste",
                "n_recomendacoes": 5
            }
            
            resposta = self.client.post(
                '/prever',
                data=json.dumps(entrada_teste),
                content_type='application/json'
            )
            
            dados = json.loads(resposta.data.decode())
            
            self.assertEqual(resposta.status_code, 200)
            self.assertIn('recomendacoes', dados)
            self.assertIn('metadata', dados)
            self.assertEqual(len(dados['recomendacoes']), 5)
            
        except Exception as e:
            logger.error(f"Erro no teste de previsão: {str(e)}")
            raise

    def test_endpoint_previsao_sem_usuario(self):
        """Testa o endpoint de previsão sem usuário."""
        logger.info("Testando endpoint de previsão - sem usuário")
        try:
            entrada_teste = {
                "n_recomendacoes": 5
            }
            
            resposta = self.client.post(
                '/prever',
                data=json.dumps(entrada_teste),
                content_type='application/json'
            )
            
            self.assertEqual(resposta.status_code, 400)
            dados = json.loads(resposta.data.decode())
            self.assertIn('erro', dados)
            self.assertIn('campos_faltantes', dados)
            
        except Exception as e:
            logger.error(f"Erro no teste de previsão sem usuário: {str(e)}")
            raise

    def test_endpoint_previsao_dados_invalidos(self):
        """Testa o endpoint de previsão com dados inválidos."""
        logger.info("Testando endpoint de previsão - dados inválidos")
        try:
            entrada_teste = {
                "id_usuario": 123,  # Deve ser string
                "n_recomendacoes": "5"  # Deve ser inteiro
            }
            
            resposta = self.client.post(
                '/prever',
                data=json.dumps(entrada_teste),
                content_type='application/json'
            )
            
            self.assertEqual(resposta.status_code, 400)
            dados = json.loads(resposta.data.decode())
            self.assertIn('erro', dados)
            
        except Exception as e:
            logger.error(f"Erro no teste de dados inválidos: {str(e)}")
            raise

    def test_rota_inexistente(self):
        """Testa acesso a rota inexistente."""
        logger.info("Testando rota inexistente")
        try:
            resposta = self.client.get('/rota_inexistente')
            self.assertEqual(resposta.status_code, 404)
            
        except Exception as e:
            logger.error(f"Erro no teste de rota inexistente: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main()'''

import unittest
import json
from src.api.app import app
import pandas as pd
import tempfile
import os
from src.modelo.recomendador import RecomendadorHibrido

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuração inicial para todos os testes."""
        app.config['TESTING'] = True
        cls.client = app.test_client()
        
        # Criar dados e modelo de teste
        cls._criar_modelo_teste()

    @classmethod
    def _criar_modelo_teste(cls):
        """Cria um modelo de teste e salva."""
        # Criar dados de exemplo
        dados_treino = pd.DataFrame({
            'idUsuario': ['user1', 'user2'],
            'historico': [['item1', 'item2'], ['item3', 'item4']],
            'historicoTimestamp': [[1000, 2000], [3000, 4000]]
        })
        
        dados_itens = pd.DataFrame({
            'Pagina': ['item1', 'item2', 'item3', 'item4'],
            'Titulo': ['Título 1', 'Título 2', 'Título 3', 'Título 4'],
            'Corpo': ['Corpo 1', 'Corpo 2', 'Corpo 3', 'Corpo 4'],
            'DataPublicacao': pd.date_range('2023-01-01', periods=4)
        })
        
        # Treinar e salvar modelo
        modelo = RecomendadorHibrido()
        modelo.treinar(dados_treino, dados_itens)
        modelo.salvar_modelo('modelos/modelos_salvos/modelo_teste')

    def test_health_check(self):
        """Testa o endpoint de verificação de saúde."""
        resposta = self.client.get('/saude')
        dados = json.loads(resposta.data.decode())
        
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(dados['status'], 'saudavel')
        self.assertIn('versao_modelo', dados)

    def test_predict_endpoint_success(self):
        """Testa o endpoint de previsão com dados válidos."""
        entrada_teste = {
            "id_usuario": "user1",
            "n_recomendacoes": 5
        }
        
        resposta = self.client.post(
            '/prever',
            data=json.dumps(entrada_teste),
            content_type='application/json'
        )
        
        dados = json.loads(resposta.data.decode())
        
        self.assertEqual(resposta.status_code, 200)
        self.assertIn('recomendacoes', dados)
        self.assertEqual(len(dados['recomendacoes']), 5)
        self.assertIn('metadata', dados)

    def test_predict_endpoint_invalid_input(self):
        """Testa o endpoint de previsão com dados inválidos."""
        casos_teste = [
            ({}, "Campos obrigatórios faltando"),
            ({"n_recomendacoes": 5}, "Campos obrigatórios faltando"),
            ({"id_usuario": 123}, "Tipo inválido"),
            ({"id_usuario": "user1", "n_recomendacoes": "5"}, "Tipo inválido")
        ]
        
        for entrada, erro_esperado in casos_teste:
            resposta = self.client.post(
                '/prever',
                data=json.dumps(entrada),
                content_type='application/json'
            )
            
            self.assertEqual(resposta.status_code, 400)
            dados = json.loads(resposta.data.decode())
            self.assertIn('erro', dados)
            self.assertIn(erro_esperado, str(dados['erro']))

    def test_route_not_found(self):
        """Testa acesso a rota inexistente."""
        resposta = self.client.get('/rota_inexistente')
        self.assertEqual(resposta.status_code, 404)
        
        dados = json.loads(resposta.data.decode())
        self.assertIn('erro', dados)
        self.assertEqual(dados['erro'], 'Rota não encontrada')

if __name__ == '__main__':
    unittest.main()