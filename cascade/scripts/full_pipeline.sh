#!/bin/bash

set -e  # Detener si hay errores

echo "=========================================="
echo "Pipeline completo de entrenamiento Cascade"
echo "=========================================="

# Paso 1: Generar listas
bash /workspace/scripts/generate_lists.sh

# Verificar que se generaron archivos
if [ ! -f /workspace/data/pos.txt ] || [ ! -f /workspace/data/neg.txt ]; then
    echo "Error: No se pudieron generar las listas de imágenes"
    exit 1
fi

# Paso 2: Entrenar
bash /workspace/scripts/train.sh

echo "=========================================="
echo "Pipeline completado exitosamente"
echo "=========================================="