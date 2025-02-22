#!/bin/bash

# Verificar API
if ! curl -f http://localhost:8000/saude > /dev/null 2>&1; then
    echo "API não está respondendo"
    exit 1
fi

# Verificar MLflow
if ! curl -f http://localhost:5000 > /dev/null 2>&1; then
    echo "MLflow não está respondendo"
    exit 1
fi

echo "Serviços saudáveis"
exit 0