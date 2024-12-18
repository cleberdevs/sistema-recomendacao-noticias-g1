import unittest
import json
from src.api.app import app

class TesteAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def teste_verificar_saude(self):
        resposta = self.app.get('/saude')
        dados = json.loads(resposta.data.decode())
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(dados['status'], 'saudavel')

    def teste_endpoint_previsao(self):
        entrada_teste = {
            "id_usuario": "usuario_teste",
            "n_recomendacoes": 5
        }
        
        resposta = self.app.post(
            '/prever',
            data=json.dumps(entrada_teste),
            content_type='application/json'
        )
        
        dados = json.loads(resposta.data.decode())
        self.assertEqual(resposta.status_code, 200)
        self.assertIn('recomendacoes', dados)

if __name__ == '__main__':
    unittest.main()
