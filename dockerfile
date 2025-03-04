FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

# Definir diretório de trabalho
WORKDIR /app

# Copiar todo o conteúdo da aplicação para o container
COPY . /app/

# Instalar Python, Java e outras dependências
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    default-jdk \
    curl \
    net-tools \
    netcat \
    sqlite3 \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Configurar variáveis de ambiente Java
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Atualizar requirements.txt para versões compatíveis
RUN sed -i 's/flask>=3.1.0/flask>=2.2.0/g' requirements.txt \
    && sed -i 's/Werkzeug>=3.1.0/Werkzeug>=2.2.0/g' requirements.txt 

# Instalar dependências Python manualmente com versões compatíveis
RUN python -m pip install --upgrade pip \
    && python -m pip install \
    tensorflow==2.5.0 \
    flask==2.2.3 \
    pandas==1.3.5 \
    numpy==1.19.5 \
    pyspark==3.2.3 \
    pyarrow==7.0.0 \
    mlflow==2.3.0 \
    python-dotenv==0.19.2 \
    scikit-learn==0.24.2 \
    nltk==3.8.1 \
    flask-restx==1.1.0 \
    Werkzeug==2.2.3 \
    swagger-ui-bundle==0.0.9

# Baixar dados NLTK
RUN python -m nltk.downloader stopwords

# Criar diretórios necessários
RUN mkdir -p dados/brutos/itens \
    dados/processados \
    modelos/modelos_salvos \
    logs \
    mlflow-artifacts \
    spark-logs \
    checkpoints \
    mlruns \
    && chmod -R 777 dados \
    && chmod -R 777 modelos \
    && chmod -R 777 logs \
    && chmod -R 777 mlflow-artifacts \
    && chmod -R 777 spark-logs \
    && chmod -R 777 checkpoints \
    && chmod -R 777 mlruns

# Tornar os scripts executáveis
RUN chmod +x scripts/setup_environment.sh \
    scripts/start_mlflow.sh \
    scripts/start_api.sh

# Criar patch para o problema do pickle usando um arquivo Python
RUN echo '#!/usr/bin/env python3' > /app/fix_recomendador.py && \
    echo 'import os' >> /app/fix_recomendador.py && \
    echo 'import sys' >> /app/fix_recomendador.py && \
    echo 'import logging' >> /app/fix_recomendador.py && \
    echo 'logging.basicConfig(level=logging.INFO)' >> /app/fix_recomendador.py && \
    echo 'logger = logging.getLogger(__name__)' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo 'def fix_file():' >> /app/fix_recomendador.py && \
    echo '    """Fix recomendador.py file to properly save/load models"""' >> /app/fix_recomendador.py && \
    echo '    file_path = "/app/src/modelo/recomendador.py"' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    # Verify file exists' >> /app/fix_recomendador.py && \
    echo '    if not os.path.exists(file_path):' >> /app/fix_recomendador.py && \
    echo '        logger.error(f"File not found: {file_path}")' >> /app/fix_recomendador.py && \
    echo '        return False' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    # Create a backup' >> /app/fix_recomendador.py && \
    echo '    backup_path = f"{file_path}.bak"' >> /app/fix_recomendador.py && \
    echo '    try:' >> /app/fix_recomendador.py && \
    echo '        with open(file_path, "r") as f:' >> /app/fix_recomendador.py && \
    echo '            original_content = f.read()' >> /app/fix_recomendador.py && \
    echo '        ' >> /app/fix_recomendador.py && \
    echo '        with open(backup_path, "w") as f:' >> /app/fix_recomendador.py && \
    echo '            f.write(original_content)' >> /app/fix_recomendador.py && \
    echo '        logger.info(f"Backup created at {backup_path}")' >> /app/fix_recomendador.py && \
    echo '    except Exception as e:' >> /app/fix_recomendador.py && \
    echo '        logger.error(f"Error creating backup: {str(e)}")' >> /app/fix_recomendador.py && \
    echo '        return False' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    # Define the new implementation for save_model method' >> /app/fix_recomendador.py && \
    echo '    new_save_method = """    def salvar_modelo(self, caminho):\n        """Salva o modelo em disco usando método nativo do TensorFlow."""\n        logger.info(f"Salvando modelo em {caminho}")\n        try:\n            # Criar diretório se não existir\n            os.makedirs(os.path.dirname(caminho), exist_ok=True)\n            \n            # 1. Salvar o modelo TensorFlow separadamente\n            model_dir = f"{caminho}_tf"\n            self.modelo.save(model_dir)\n            logger.info(f"Modelo TensorFlow salvo em {model_dir}")\n            \n            # 2. Salvar outros componentes separadamente\n            import pickle\n            \n            # Componentes para salvar (exceto o modelo)\n            componentes = {\n                \'tfidf\': self.tfidf,\n                \'item_id_to_index\': self.item_id_to_index,\n                \'index_to_item_id\': self.index_to_item_id,\n                \'usuario_id_to_index\': self.usuario_id_to_index,\n                \'index_to_usuario_id\': self.index_to_usuario_id,\n                \'item_count\': self.item_count,\n                \'itens_usuario\': self.itens_usuario,\n                \'features_item\': self.features_item,\n                \'dim_embedding\': self.dim_embedding,\n                \'dim_features_texto\': self.dim_features_texto\n            }\n            \n            # Salvar em arquivo separado\n            with open(f"{caminho}_componentes.pkl", "wb") as f:\n                pickle.dump(componentes, f)\n            logger.info(f"Componentes auxiliares salvos em {caminho}_componentes.pkl")\n            \n            # 3. Criar arquivo de marcação para indicar que o modelo foi salvo corretamente\n            with open(f"{caminho}_success", "w") as f:\n                f.write("1")\n            \n            logger.info("Modelo salvo com sucesso no formato separado")\n            \n            # 4. Registrar no MLflow quando disponível\n            try:\n                if hasattr(self, \'mlflow_config\') and self.mlflow_config:\n                    self.mlflow_config.log_artefato(model_dir)\n                    self.mlflow_config.log_artefato(f"{caminho}_componentes.pkl")\n                    logger.info("Artefatos registrados no MLflow")\n            except Exception as e:\n                logger.warning(f"Aviso ao registrar no MLflow: {str(e)}")\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f"Erro ao salvar modelo: {str(e)}")\n            return False\n"""' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    # Define the new implementation for load_model method' >> /app/fix_recomendador.py && \
    echo '    new_load_method = """    @classmethod\n    def carregar_modelo(cls, caminho):\n        """Carrega um modelo salvo do disco."""\n        logger.info(f"Carregando modelo de {caminho}")\n        try:\n            # Verificar se é o formato novo (com arquivos separados)\n            if os.path.exists(f"{caminho}_success"):\n                logger.info("Detectado formato de salvamento separado")\n                \n                # 1. Carregar componentes\n                import pickle\n                with open(f"{caminho}_componentes.pkl", "rb") as f:\n                    componentes = pickle.load(f)\n                \n                # 2. Carregar modelo TensorFlow\n                import tensorflow as tf\n                model_dir = f"{caminho}_tf"\n                modelo_tf = tf.keras.models.load_model(model_dir)\n                \n                # 3. Criar instância\n                instancia = cls(\n                    dim_embedding=componentes["dim_embedding"],\n                    dim_features_texto=componentes["dim_features_texto"]\n                )\n                \n                # 4. Atribuir componentes\n                instancia.modelo = modelo_tf\n                instancia.tfidf = componentes["tfidf"]\n                instancia.item_id_to_index = componentes["item_id_to_index"]\n                instancia.index_to_item_id = componentes["index_to_item_id"]\n                instancia.usuario_id_to_index = componentes["usuario_id_to_index"]\n                instancia.index_to_usuario_id = componentes["index_to_usuario_id"]\n                instancia.item_count = componentes["item_count"]\n                instancia.itens_usuario = componentes["itens_usuario"]\n                instancia.features_item = componentes["features_item"]\n                \n                logger.info("Modelo carregado com sucesso (formato novo)")\n                return instancia\n            \n            else:\n                # Tentar formato antigo (arquivo único)\n                logger.warning("Tentando carregar modelo no formato antigo (pickle)")\n                try:\n                    import pickle\n                    with open(caminho, "rb") as f:\n                        dados_modelo = pickle.load(f)\n                    \n                    instancia = cls(\n                        dim_embedding=dados_modelo["dim_embedding"],\n                        dim_features_texto=dados_modelo["dim_features_texto"]\n                    )\n                    \n                    instancia.modelo = dados_modelo["modelo"]\n                    instancia.tfidf = dados_modelo["tfidf"]\n                    instancia.item_id_to_index = dados_modelo["item_id_to_index"]\n                    instancia.index_to_item_id = dados_modelo["index_to_item_id"]\n                    instancia.usuario_id_to_index = dados_modelo["usuario_id_to_index"]\n                    instancia.index_to_usuario_id = dados_modelo["index_to_usuario_id"]\n                    instancia.item_count = dados_modelo["item_count"]\n                    instancia.itens_usuario = dados_modelo["itens_usuario"]\n                    instancia.features_item = dados_modelo["features_item"]\n                    \n                    logger.info("Modelo carregado com sucesso (formato antigo)")\n                    return instancia\n                except Exception as e:\n                    logger.error(f"Erro ao carregar modelo no formato antigo: {str(e)}")\n                    raise\n            \n        except Exception as e:\n            logger.error(f"Erro ao carregar modelo: {str(e)}")\n            raise\n"""' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    try:' >> /app/fix_recomendador.py && \
    echo '        # Read the original file' >> /app/fix_recomendador.py && \
    echo '        lines = []' >> /app/fix_recomendador.py && \
    echo '        with open(file_path, "r") as f:' >> /app/fix_recomendador.py && \
    echo '            lines = f.readlines()' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Find the methods to replace' >> /app/fix_recomendador.py && \
    echo '        save_start_idx = -1' >> /app/fix_recomendador.py && \
    echo '        save_end_idx = -1' >> /app/fix_recomendador.py && \
    echo '        load_start_idx = -1' >> /app/fix_recomendador.py && \
    echo '        load_end_idx = -1' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Find salvar_modelo method' >> /app/fix_recomendador.py && \
    echo '        for i, line in enumerate(lines):' >> /app/fix_recomendador.py && \
    echo '            if "def salvar_modelo(self, caminho):" in line:' >> /app/fix_recomendador.py && \
    echo '                save_start_idx = i' >> /app/fix_recomendador.py && \
    echo '                # Find the end (next method or end of class)' >> /app/fix_recomendador.py && \
    echo '                for j in range(i + 1, len(lines)):' >> /app/fix_recomendador.py && \
    echo '                    if lines[j].strip().startswith("def ") or lines[j].strip().startswith("@classmethod"):' >> /app/fix_recomendador.py && \
    echo '                        save_end_idx = j' >> /app/fix_recomendador.py && \
    echo '                        break' >> /app/fix_recomendador.py && \
    echo '                break' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Find carregar_modelo method' >> /app/fix_recomendador.py && \
    echo '        for i, line in enumerate(lines):' >> /app/fix_recomendador.py && \
    echo '            if "def carregar_modelo(cls, caminho):" in line and "@classmethod" in lines[i-1]:' >> /app/fix_recomendador.py && \
    echo '                load_start_idx = i - 1  # Include the @classmethod line' >> /app/fix_recomendador.py && \
    echo '                # Find the end (next method or end of class)' >> /app/fix_recomendador.py && \
    echo '                for j in range(i + 1, len(lines)):' >> /app/fix_recomendador.py && \
    echo '                    if lines[j].strip().startswith("def ") or lines[j].strip().startswith("@classmethod") or lines[j].strip().startswith("class "):' >> /app/fix_recomendador.py && \
    echo '                        load_end_idx = j' >> /app/fix_recomendador.py && \
    echo '                        break' >> /app/fix_recomendador.py && \
    echo '                if load_end_idx == -1:  # If no next method found, use end of file' >> /app/fix_recomendador.py && \
    echo '                    load_end_idx = len(lines)' >> /app/fix_recomendador.py && \
    echo '                break' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Ensure we found the methods' >> /app/fix_recomendador.py && \
    echo '        if save_start_idx == -1 or save_end_idx == -1:' >> /app/fix_recomendador.py && \
    echo '            logger.error("Could not find salvar_modelo method")' >> /app/fix_recomendador.py && \
    echo '            return False' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        if load_start_idx == -1 or load_end_idx == -1:' >> /app/fix_recomendador.py && \
    echo '            logger.error("Could not find carregar_modelo method")' >> /app/fix_recomendador.py && \
    echo '            return False' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Replace the methods' >> /app/fix_recomendador.py && \
    echo '        new_lines = []' >> /app/fix_recomendador.py && \
    echo '        for i, line in enumerate(lines):' >> /app/fix_recomendador.py && \
    echo '            if i == save_start_idx:' >> /app/fix_recomendador.py && \
    echo '                # Add the new save method' >> /app/fix_recomendador.py && \
    echo '                new_lines.append(new_save_method)' >> /app/fix_recomendador.py && \
    echo '                # Skip the old method lines' >> /app/fix_recomendador.py && \
    echo '                i = save_end_idx - 1' >> /app/fix_recomendador.py && \
    echo '            elif i == load_start_idx:' >> /app/fix_recomendador.py && \
    echo '                # Add the new load method' >> /app/fix_recomendador.py && \
    echo '                new_lines.append(new_load_method)' >> /app/fix_recomendador.py && \
    echo '                # Skip the old method lines' >> /app/fix_recomendador.py && \
    echo '                i = load_end_idx - 1' >> /app/fix_recomendador.py && \
    echo '            elif save_start_idx <= i < save_end_idx or load_start_idx <= i < load_end_idx:' >> /app/fix_recomendador.py && \
    echo '                # Skip these lines as they are being replaced' >> /app/fix_recomendador.py && \
    echo '                continue' >> /app/fix_recomendador.py && \
    echo '            else:' >> /app/fix_recomendador.py && \
    echo '                new_lines.append(line)' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        # Write the modified content back to the file' >> /app/fix_recomendador.py && \
    echo '        with open(file_path, "w") as f:' >> /app/fix_recomendador.py && \
    echo '            f.writelines(new_lines)' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '        logger.info("Successfully replaced methods in the file")' >> /app/fix_recomendador.py && \
    echo '        return True' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo '    except Exception as e:' >> /app/fix_recomendador.py && \
    echo '        logger.error(f"Error replacing methods: {str(e)}")' >> /app/fix_recomendador.py && \
    echo '        # Restore from backup on error' >> /app/fix_recomendador.py && \
    echo '        try:' >> /app/fix_recomendador.py && \
    echo '            with open(backup_path, "r") as f:' >> /app/fix_recomendador.py && \
    echo '                backup_content = f.read()' >> /app/fix_recomendador.py && \
    echo '            with open(file_path, "w") as f:' >> /app/fix_recomendador.py && \
    echo '                f.write(backup_content)' >> /app/fix_recomendador.py && \
    echo '            logger.info("Restored original file from backup after error")' >> /app/fix_recomendador.py && \
    echo '        except Exception as restore_error:' >> /app/fix_recomendador.py && \
    echo '            logger.error(f"Error restoring backup: {str(restore_error)}")' >> /app/fix_recomendador.py && \
    echo '        return False' >> /app/fix_recomendador.py && \
    echo '' >> /app/fix_recomendador.py && \
    echo 'if __name__ == "__main__":' >> /app/fix_recomendador.py && \
    echo '    success = fix_file()' >> /app/fix_recomendador.py && \
    echo '    if success:' >> /app/fix_recomendador.py && \
    echo '        print("✅ Correção aplicada com sucesso")' >> /app/fix_recomendador.py && \
    echo '        sys.exit(0)' >> /app/fix_recomendador.py && \
    echo '    else:' >> /app/fix_recomendador.py && \
    echo '        print("❌ Falha ao aplicar correção")' >> /app/fix_recomendador.py && \
    echo '        sys.exit(1)' >> /app/fix_recomendador.py

# Tornar o script executável e testá-lo (mas continuar mesmo que falhe no build)
RUN chmod +x /app/fix_recomendador.py && \
    python /app/fix_recomendador.py || echo "AVISO: Correção será tentada no runtime"

# Script de inicialização para aplicar a correção em runtime se necessário
RUN echo '#!/bin/bash' > /app/init.sh && \
    echo 'set -e' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo '# Aplicar correção no início' >> /app/init.sh && \
    echo 'echo "Verificando e aplicando correção para problema do pickle..."' >> /app/init.sh && \
    echo 'python /app/fix_recomendador.py' >> /app/init.sh && \
    echo 'if [ $? -ne 0 ]; then' >> /app/init.sh && \
    echo '    echo "AVISO: Falha na correção automática. Pipeline pode falhar ao salvar/carregar modelos."' >> /app/init.sh && \
    echo 'else' >> /app/init.sh && \
    echo '    echo "✓ Correção aplicada com sucesso"' >> /app/init.sh && \
    echo 'fi' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo '# Iniciar o MLflow server' >> /app/init.sh && \
    echo 'echo "Iniciando MLflow server..."' >> /app/init.sh && \
    echo 'mkdir -p mlflow-artifacts logs mlruns' >> /app/init.sh && \
    echo 'nohup mlflow server \\' >> /app/init.sh && \
    echo '    --host 0.0.0.0 \\' >> /app/init.sh && \
    echo '    --port 5000 \\' >> /app/init.sh && \
    echo '    --backend-store-uri "file:///app/mlruns" \\' >> /app/init.sh && \
    echo '    --default-artifact-root "/app/mlflow-artifacts" > logs/mlflow.log 2>&1 &' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo '# Verificar se MLflow está rodando' >> /app/init.sh && \
    echo 'echo "Aguardando MLflow iniciar (30s)..."' >> /app/init.sh && \
    echo 'sleep 5' >> /app/init.sh && \
    echo 'for i in {1..25}; do' >> /app/init.sh && \
    echo '    if nc -z localhost 5000; then' >> /app/init.sh && \
    echo '        echo "MLflow iniciado com sucesso na porta 5000"' >> /app/init.sh && \
    echo '        break' >> /app/init.sh && \
    echo '    fi' >> /app/init.sh && \
    echo '    echo "Tentativa $i: MLflow ainda não está rodando..."' >> /app/init.sh && \
    echo '    sleep 1' >> /app/init.sh && \
    echo '    if [ $i -eq 25 ]; then' >> /app/init.sh && \
    echo '        echo "Falha ao iniciar MLflow após 30 segundos"' >> /app/init.sh && \
    echo '        cat logs/mlflow.log' >> /app/init.sh && \
    echo '        exit 1' >> /app/init.sh && \
    echo '    fi' >> /app/init.sh && \
    echo 'done' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo '# Configure environment variables' >> /app/init.sh && \
    echo 'export MLFLOW_TRACKING_URI="http://localhost:5000"' >> /app/init.sh && \
    echo 'export FLASK_APP=src.api.app:app' >> /app/init.sh && \
    echo 'export FLASK_ENV=development' >> /app/init.sh && \
    echo 'export PYTHONPATH=/app:${PYTHONPATH:-}' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo 'echo "MLflow está pronto!"' >> /app/init.sh && \
    echo 'echo "IMPORTANTE: Para executar o pipeline corretamente, use:"' >> /app/init.sh && \
    echo 'echo "docker exec recomendador /app/run_pipeline.sh"' >> /app/init.sh && \
    echo 'echo "Para iniciar a API: docker exec recomendador ./scripts/start_api.sh"' >> /app/init.sh && \
    echo '' >> /app/init.sh && \
    echo '# Manter o container rodando' >> /app/init.sh && \
    echo 'tail -f /dev/null' >> /app/init.sh

# Script para executar o pipeline com correção garantida
RUN echo '#!/bin/bash' > /app/run_pipeline.sh && \
    echo 'set -e' >> /app/run_pipeline.sh && \
    echo '' >> /app/run_pipeline.sh && \
    echo 'cd /app' >> /app/run_pipeline.sh && \
    echo '' >> /app/run_pipeline.sh && \
    echo '# Garantir que a correção foi aplicada' >> /app/run_pipeline.sh && \
    echo 'python /app/fix_recomendador.py' >> /app/run_pipeline.sh && \
    echo 'if [ $? -ne 0 ]; then' >> /app/run_pipeline.sh && \
    echo '    echo "ERRO: Não foi possível aplicar a correção para o problema do pickle"' >> /app/run_pipeline.sh && \
    echo '    echo "O pipeline pode falhar ao tentar salvar o modelo"' >> /app/run_pipeline.sh && \
    echo '    read -p "Deseja continuar mesmo assim? (s/n): " response' >> /app/run_pipeline.sh && \
    echo '    if [[ "$response" != "s" ]]; then' >> /app/run_pipeline.sh && \
    echo '        echo "Operação cancelada pelo usuário"' >> /app/run_pipeline.sh && \
    echo '        exit 1' >> /app/run_pipeline.sh && \
    echo '    fi' >> /app/run_pipeline.sh && \
    echo 'fi' >> /app/run_pipeline.sh && \
    echo '' >> /app/run_pipeline.sh && \
    echo '# Configurar variáveis de ambiente' >> /app/run_pipeline.sh && \
    echo 'export MLFLOW_TRACKING_URI="http://localhost:5000"' >> /app/run_pipeline.sh && \
    echo 'export PYTHONPATH=/app:${PYTHONPATH:-}' >> /app/run_pipeline.sh && \
    echo '' >> /app/run_pipeline.sh && \
    echo '# Executar o pipeline' >> /app/run_pipeline.sh && \
    echo 'echo "Executando pipeline..."' >> /app/run_pipeline.sh && \
    echo 'python pipeline.py' >> /app/run_pipeline.sh && \
    echo 'exit_code=$?' >> /app/run_pipeline.sh && \
    echo 'if [ $exit_code -ne 0 ]; then' >> /app/run_pipeline.sh && \
    echo '    echo "ERRO: Pipeline falhou com código $exit_code"' >> /app/run_pipeline.sh && \
    echo '    exit $exit_code' >> /app/run_pipeline.sh && \
    echo 'fi' >> /app/run_pipeline.sh && \
    echo '' >> /app/run_pipeline.sh && \
    echo 'echo "Pipeline executado com sucesso!"' >> /app/run_pipeline.sh

# Tornar os scripts executáveis
RUN chmod +x /app/init.sh /app/run_pipeline.sh

# Expor portas
EXPOSE 5000 8000

# Definir volumes
VOLUME ["/app/dados", "/app/logs", "/app/mlflow-artifacts", "/app/modelos/modelos_salvos"]

# Configurações do Spark para o Docker
ENV SPARK_LOCAL_IP="127.0.0.1"
ENV SPARK_LOCAL_DIRS="/tmp"
ENV SPARK_WORKER_DIR="/tmp"
ENV PYSPARK_PYTHON="/usr/bin/python3"
ENV PYSPARK_DRIVER_PYTHON="/usr/bin/python3"
ENV PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g pyspark-shell"
ENV SPARK_DAEMON_MEMORY="2g"
ENV SPARK_DRIVER_MEMORY="4g"
ENV SPARK_EXECUTOR_MEMORY="4g"

# Definir o script de entrada
ENTRYPOINT ["/app/init.sh"]