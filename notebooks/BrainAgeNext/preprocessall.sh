#!/bin/bash

# --- Configuración ---
INPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/ADNI/ADNI_3_T1/brainagenext/niftis_malos_prepreprocessed"   # Carpeta con archivos NIfTI (cerebro ya extraído)
OUTPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/ADNI/ADNI_3_T1/brainagenext/niftis_malos_preprocesed"  # Carpeta de salida
MNI_TEMPLATE="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"  # Ruta a plantilla MNI152

mkdir -p "$OUTPUT_DIR"

# --- Procesamiento de cada archivo ---
for img in "$INPUT_DIR"/*.nii; do
    filename=$(basename "$img")                     # e.g., sujeto.nii
    base="${filename%%.*}"                          # e.g., sujeto

    echo "Procesando $base..."

    # 1. Skull stripping (aunque ya esté extraído, puede mejorar resultados)
    bet "$img" "$OUTPUT_DIR/${base}_brain.nii.gz" -R -f 0.5 -g 0

    # 2. Registro usando la imagen con cerebro extraído
    antsRegistrationSyNQuick.sh \
        -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$OUTPUT_DIR/${base}_brain.nii.gz" \
        -o "$OUTPUT_DIR/${base}_to_MNI_"

    # 3. N4 bias field correction sobre la imagen registrada
    N4BiasFieldCorrection -d 3 \
        -i "$OUTPUT_DIR/${base}_to_MNI_Warped.nii.gz" \
        -o "$OUTPUT_DIR/${base}_final_preproc.nii.gz"

    echo "✓ $base procesado."
done

echo "✅ Todos los archivos fueron preprocesados y guardados en $OUTPUT_DIR"

