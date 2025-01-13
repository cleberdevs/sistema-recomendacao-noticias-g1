import unittest
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
    unittest.main()