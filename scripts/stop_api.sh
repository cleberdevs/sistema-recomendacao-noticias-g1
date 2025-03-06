#!/bin/bash

# Define a porta do Flask (ajuste conforme necessário)
PORT=8000

# Encontra o PID do processo Flask rodando na porta especificada
PID=$(lsof -t -i :$PORT)

# Verifica se o processo foi encontrado
if [ -z "$PID" ]; then
  echo "Nenhum processo Flask encontrado na porta $PORT."
else
  echo "Parando o Flask (PID: $PID) na porta $PORT..."
  kill $PID
  # Verifica se o processo foi parado com sucesso
  if [ $? -eq 0 ]; then
    echo "Flask parado com sucesso."
  else
    echo "Falha ao parar o Flask."
  fi
fi