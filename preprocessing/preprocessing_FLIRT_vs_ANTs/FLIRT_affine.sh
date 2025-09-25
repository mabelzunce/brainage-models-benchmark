#!/bin/bash

INPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Raw T1/datos_check_de_preprocesamiento"
OUTPUT_DIR="${INPUT_DIR%/}/preprocessed_FLIRT_affine"
mkdir -p "$OUTPUT_DIR"

# Ruta al template MNI (ajústalo si usás otra versión de FSL)
MNI_TEMPLATE="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain"

# Recorre los archivos .nii y .nii.gz
shopt -s nullglob
for nii in "$INPUT_DIR"/*.nii "$INPUT_DIR"/*.nii.gz; do
    base=$(basename "$nii")
    base_noext="${base%%.*}"

    echo "Procesando: $base"

    # 1. N4 Bias Field Correction
    N4_OUT="${OUTPUT_DIR}/${base_noext}_n4.nii.gz"
    N4BiasFieldCorrection -i "$nii" -o "$N4_OUT"

    # 2. Skull stripping con SynthStrip
    STRIP_OUT="${OUTPUT_DIR}/${base_noext}_brain.nii.gz"
    MASK_OUT="${OUTPUT_DIR}/${base_noext}_mask.mgz"
    mri_synthstrip -i "$N4_OUT" -o "$STRIP_OUT" -m "$MASK_OUT"

    # 3. Registro rígido a MNI
    FLIRT_OUT="${OUTPUT_DIR}/${base_noext}_MNI.nii.gz"
    MAT_OUT="${OUTPUT_DIR}/${base_noext}_to_MNI.mat"
    flirt -in "$STRIP_OUT" -ref "$MNI_TEMPLATE" -out "$FLIRT_OUT" -omat "$MAT_OUT" -dof 12

done

echo "✅ Procesamiento completado en: $OUTPUT_DIR"