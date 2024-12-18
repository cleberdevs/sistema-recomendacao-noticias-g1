import unittest
import pandas as pd
import numpy as np
from src.modelo.recomendador import RecomendadorHibrido
import mlflow

class TesteRecomendador(unittest.TestCase):
    def setUp(self):
        self.modelo = RecomendadorHibrido()
        
        # Criar dados de exemplo
        self.dados_treino = pd.DataFrame({
            'idUsuario': ['usuario1', 'usuario2'],
            'historico': [['item1', 'item2'], ['item3', 'item4']]
        })
        
        self.dados_itens = pd.DataFrame({
            'Pagina': ['item1', 'item2', 'item3', 'item4'],
            'Titulo': ['Título 1', 'Título 2', 'Título 3', 'Título 4'],
            'Corpo': ['Corpo 1', 'Corpo 2', 'Corpo 3', 'Corpo 4'],
            'DataPublicacao': ['2023-01-01'] * 4
        })

    def teste_inicializacao_modelo(self):
        self.assertIsNotNone(self.modelo)
        self.assertIsNotNone(self.modelo.mlflow_config)

    def teste_treinamento_modelo(self):
        with mlflow.start_run():
            self.modelo.treinar(self.dados_treino, self.dados_itens)
            self.assertIsNotNone(self.modelo.matriz_similaridade)

    def teste_previsao_modelo(self):
        with mlflow.start_run():
            self.modelo.treinar(self.dados_treino, self.dados_itens)
            previsoes = self.modelo.prever('usuario1', n_recomendacoes=2)
            self.assertEqual(len(previsoes), 2)

if __name__ == '__main__':
    unittest.main()
