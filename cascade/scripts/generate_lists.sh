#!/bin/bash

echo "=========================================="
echo "  GENERANDO LISTAS DE IMÁGENES"
echo "=========================================="

# Limpiar archivos anteriores
rm -f /cascade/data/pos.txt /cascade/data/neg.txt

# ================= NEGATIVOS =================
echo ""
echo "=== Imágenes negativas ==="

# Buscar directamente en /cascade/negativos (NO en /cascade/data/negativos)
find /cascade/negativos -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) > /cascade/data/neg.txt

NUM_NEG=$(wc -l < /cascade/data/neg.txt)
echo "Encontradas: $NUM_NEG imágenes"

# ================= POSITIVOS =================
echo ""
echo "=== Imágenes positivas ==="

# Buscar directamente en /cascade/Personas (NO en /cascade/data/Personas)
find /cascade/Personas -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | \
    awk '{print $0 " 1 0 0 64 128"}' > /cascade/data/pos.txt

NUM_POS=$(wc -l < /cascade/data/pos.txt)
echo "Encontradas: $NUM_POS imágenes"

# ================= VERIFICACIÓN =================
echo ""
echo "=========================================="
echo "  VERIFICACIÓN"
echo "=========================================="

if [ $NUM_POS -eq 0 ]; then
    echo "❌ ERROR: No se encontraron imágenes positivas"
    echo "Verificando estructura de directorios..."
    ls -la /cascade/
    echo ""
    echo "Contenido de /cascade/Personas:"
    ls /cascade/Personas/ | head -5
    exit 1
fi

if [ $NUM_NEG -eq 0 ]; then
    echo "❌ ERROR: No se encontraron imágenes negativas"
    echo "Verificando estructura de directorios..."
    ls -la /cascade/
    echo ""
    echo "Contenido de /cascade/negativos:"
    ls /cascade/negativos/ | head -5
    exit 1
fi

# Verificar primera imagen positiva
first_pos=$(head -1 /cascade/data/pos.txt | cut -d' ' -f1)
echo ""
echo "Primera imagen positiva: $first_pos"

if [ -f "$first_pos" ]; then
    echo "✓ Archivo existe"
    dimensions=$(identify -format "%wx%h" "$first_pos" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  Dimensiones: $dimensions"
    fi
else
    echo "❌ ERROR: Archivo no existe"
    echo ""
    echo "Contenido de pos.txt (primera línea):"
    head -1 /cascade/data/pos.txt
    echo ""
    echo "Archivos en /cascade/Personas:"
    ls -la /cascade/Personas/ | head -5
    exit 1
fi

# Verificar primera imagen negativa
first_neg=$(head -1 /cascade/data/neg.txt)
echo ""
echo "Primera imagen negativa: $first_neg"

if [ -f "$first_neg" ]; then
    echo "✓ Archivo existe"
else
    echo "❌ ERROR: Archivo no existe"
    echo ""
    echo "Contenido de neg.txt (primera línea):"
    head -1 /cascade/data/neg.txt
    exit 1
fi

# Verificar que no hay rutas duplicadas
echo ""
echo "Verificando rutas duplicadas..."
if grep -q "/cascade/data//cascade" /cascade/data/pos.txt; then
    echo "❌ ERROR: Se encontraron rutas duplicadas en pos.txt"
    grep "/cascade/data//cascade" /cascade/data/pos.txt | head -3
    exit 1
fi

if grep -q "/cascade/data//cascade" /cascade/data/neg.txt; then
    echo "❌ ERROR: Se encontraron rutas duplicadas en neg.txt"
    grep "/cascade/data//cascade" /cascade/data/neg.txt | head -3
    exit 1
fi

echo "✓ No hay rutas duplicadas"

# Mostrar ejemplos
echo ""
echo "=========================================="
echo "  EJEMPLOS DE ARCHIVOS GENERADOS"
echo "=========================================="

echo ""
echo "pos.txt (primeras 3 líneas):"
head -3 /cascade/data/pos.txt

echo ""
echo "neg.txt (primeras 3 líneas):"
head -3 /cascade/data/neg.txt

echo ""
echo "=========================================="
echo "  RESUMEN"
echo "=========================================="
echo "✓ Positivas: $NUM_POS"
echo "✓ Negativas: $NUM_NEG"
echo "✓ Listas generadas correctamente"
echo ""
echo "Siguiente paso: bash /cascade/scripts/train.sh"