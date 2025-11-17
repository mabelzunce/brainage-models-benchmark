#!/bin/bash

# ==========================
# USO:
#   ./copiar_stats.sh /ruta/a/subjects /ruta/a/carpeta_destino
# ==========================

SOURCE_DIR="$1"
DEST_DIR="$2"

# --- Verificaciones básicas ---
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
    echo "Uso: ./copiar_stats.sh <carpeta_subjects> <carpeta_destino>"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: La carpeta de subjects no existe: $SOURCE_DIR"
    exit 1
fi

# Crear carpeta destino si no existe
mkdir -p "$DEST_DIR"

echo "Copiando carpetas stats desde:"
echo "  $SOURCE_DIR"
echo "Hacia:"
echo "  $DEST_DIR"
echo ""

# --- Bucle por cada sujeto ---
for subj in "$SOURCE_DIR"/*; do
    if [ -d "$subj/stats" ]; then
        subj_name=$(basename "$subj")
        echo "Copiando: $subj_name"

        mkdir -p "$DEST_DIR/$subj_name"
        cp -r "$subj/stats" "$DEST_DIR/$subj_name/"
    fi
done

echo ""
echo "Copiado completado."

