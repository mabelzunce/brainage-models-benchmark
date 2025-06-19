#!/bin/bash

INPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Raw_T1/datos_check_de_preprocesamiento"
OUTPUT_DIR="${INPUT_DIR%/}/preprocessed_ANTS_rigid"
mkdir -p "$OUTPUT_DIR"

MNI_TEMPLATE="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

shopt -s nullglob

for nii in "$INPUT_DIR"/*.nii "$INPUT_DIR"/*.nii.gz; do
    base=$(basename "$nii")
    base_noext="${base%.nii.gz}"
    base_noext="${base_noext%.nii}"

    echo "🔄 Procesando: $base"

    # 1. N4 Bias Field Correction
    N4_OUT="${OUTPUT_DIR}/${base_noext}_n4.nii.gz"
    N4BiasFieldCorrection -i "$nii" -o "$N4_OUT"

    # 2. Skull stripping
    STRIP_OUT="${OUTPUT_DIR}/${base_noext}_brain.nii.gz"
    MASK_OUT="${OUTPUT_DIR}/${base_noext}_mask.mgz"
    mri_synthstrip -i "$N4_OUT" -o "$STRIP_OUT" -m "$MASK_OUT"

    # 3. Registro rígido con ANTs
    OUT_PREFIX="${OUTPUT_DIR}/${base_noext}_ANTS"
    antsRegistrationSyN.sh -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$STRIP_OUT" \
        -o "$OUT_PREFIX" \
        -t r \
        -n 8                  
done

echo "✅ Procesamiento completado en: $OUTPUT_DIR"


