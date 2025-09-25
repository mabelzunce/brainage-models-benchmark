#!/bin/bash
#paths
INPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Raw_T1/CP0173_skullstripped"
OUTPUT_DIR="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Preprocessed/DeepBrainNet/CP0173skullstripped"
mkdir -p "$OUTPUT_DIR"

MNI_TEMPLATE="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

shopt -s nullglob

for nii in "$INPUT_DIR"/*.nii.gz "$INPUT_DIR"/*.nii.gz; do
    base=$(basename "$nii")
    base_noext="${base%.nii.gz}"
    base_noext="${base_noext%.nii}"

    echo "🔄 Preprocesing: $base"

    # 1. N4 Bias Field Correction
    N4_OUT="${OUTPUT_DIR}/${base_noext}_n4.nii.gz"
    N4BiasFieldCorrection -i "$nii" -o "$N4_OUT"

    #2. Skull stripping with BET
    STRIP_OUT="${OUTPUT_DIR}/${base_noext}_brain.nii.gz"
    MASK_OUT="${OUTPUT_DIR}/${base_noext}_brain_mask.nii.gz"
    bet "$N4_OUT" "${OUTPUT_DIR}/${base_noext}_brain" -m

    # 3. Rigid registration with ANTs
    OUT_PREFIX="${OUTPUT_DIR}/${base_noext}_ANTS"
    antsRegistrationSyN.sh -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$N4_OUT" \
        -o "$OUT_PREFIX" \
        -t a \
        -n 8                 
done

echo "✅ preprocesing complete: $OUTPUT_DIR"
