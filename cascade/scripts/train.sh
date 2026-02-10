#!/usr/bin/env bash
set -euo pipefail

# Root dir (script is in scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POS_DIR="$ROOT_DIR/Personas"
NEG_DIR="$ROOT_DIR/negativos"

# Buscar pos.txt / neg.txt
for f in "$POS_DIR/pos.txt" "$ROOT_DIR/pos.txt" "$ROOT_DIR/data/pos.txt"; do
    if [ -f "$f" ]; then POS_TXT="$f"; break; fi
done
for f in "$NEG_DIR/neg.txt" "$ROOT_DIR/neg.txt" "$ROOT_DIR/data/neg.txt"; do
    if [ -f "$f" ]; then NEG_TXT="$f"; break; fi
done

if [ -z "${POS_TXT:-}" ]; then echo "Error: pos.txt no encontrado." >&2; exit 1; fi
if [ -z "${NEG_TXT:-}" ]; then echo "Error: neg.txt no encontrado." >&2; exit 1; fi

OUTPUT_DIR="$ROOT_DIR/data/output"
VEC_FILE="$OUTPUT_DIR/positives.vec"
CASCADE_DIR="$OUTPUT_DIR/cascade"

# Parámetros
NUM_STAGES=8
W=128       
H=256

mkdir -p "$OUTPUT_DIR" "$CASCADE_DIR"

echo "=== Creando vector de muestras positivas (vec) ==="
TOTAL_POS=$(wc -l < "$POS_TXT" || echo 0)
echo "Entradas en $POS_TXT: $TOTAL_POS"

if [ "$TOTAL_POS" -lt 1 ]; then 
    echo "No hay muestras positivas listadas en $POS_TXT" >&2
    exit 1
fi

# Crear .vec con TODAS las muestras disponibles (o un número alto)
# VEC_SAMPLES=$((TOTAL_POS > 7000 ? 7000 : TOTAL_POS))
VEC_SAMPLES=450

echo "Creando archivo .vec con $VEC_SAMPLES muestras..."

# Crear archivo temporal con rutas relativas
POS_TMP="$ROOT_DIR/pos_createsamples.txt"
rm -f "$POS_TMP"

while IFS= read -r line || [ -n "$line" ]; do
    img=$(echo "$line" | awk '{print $1}')
    rest=$(echo "$line" | cut -d' ' -f2-)
    img_rel="${img#/}"
    img_rel="${img_rel#cascade/}"
    echo "$img_rel $rest" >> "$POS_TMP"
done < "$POS_TXT"

opencv_createsamples -info "$POS_TMP" -vec "$VEC_FILE" -num "$VEC_SAMPLES" -w "$W" -h "$H"

rm -f "$POS_TMP"

# ===== CÁLCULO IMPORTANTE =====
# NUM_POS debe ser significativamente menor que VEC_SAMPLES
# Fórmula: NUM_POS = VEC_SAMPLES * 0.8 / (1 + 0.1 * NUM_STAGES)

# NUM_POS=$(awk "BEGIN {print int($VEC_SAMPLES * 0.8 / (1.0 + 0.1 * $NUM_STAGES))}")
NUM_POS=250

# NUM_NEG=$((NUM_POS / 2))  # Negativas: aproximadamente la mitad de positivas
NUM_NEG=893

echo ""
echo "=========================================="
echo "  CONFIGURACIÓN DEL ENTRENAMIENTO"
echo "=========================================="
echo "Muestras en .vec file: $VEC_SAMPLES"
echo "Muestras usadas por stage: $NUM_POS"
echo "Muestras negativas: $NUM_NEG"
echo "Número de stages: $NUM_STAGES"
echo "Tamaño de ventana: ${W}x${H}"
echo ""
echo "IMPORTANTE:"
echo "  - El .vec tiene $VEC_SAMPLES muestras"
echo "  - Cada stage usará $NUM_POS muestras"
echo "  - Esto permite completar los $NUM_STAGES stages"
echo "=========================================="
echo ""

read -p "¿Continuar con el entrenamiento? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Entrenamiento cancelado."
    exit 0
fi

echo "=== Iniciando entrenamiento con opencv_traincascade ==="
opencv_traincascade \
    -data "$CASCADE_DIR" \
    -vec "$VEC_FILE" \
    -bg "$NEG_TXT" \
    -numPos "$NUM_POS" \
    -numNeg "$NUM_NEG" \
    -numStages "$NUM_STAGES" \
    -w "$W" -h "$H" \
    -minHitRate 0.995 \
    -maxFalseAlarmRate 0.5 \
    -featureType LBP \
    -mode ALL \
    -precalcValBufSize 1024 \
    -precalcIdxBufSize 1024

echo ""
echo "=========================================="
echo "  ENTRENAMIENTO COMPLETADO"
echo "=========================================="
echo "Clasificador guardado en: $CASCADE_DIR/cascade.xml"