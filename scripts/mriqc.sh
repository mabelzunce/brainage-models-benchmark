#!/bin/bash

# ================================
# CONFIGURACIÓN DEL USUARIO
# ================================

bids_root_dir="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Raw_T1/sampled_imgs_ADNI"
out_root="/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Raw_T1/Image_quality/ADNI"
container_path="/home/lautaro/mriqc-0.15.1.simg"

nthreads=8
mem=16   # GB

# ================================
# LOOP SOBRE TODOS LOS SUJETOS BIDS
# ================================

echo "Buscando sujetos dentro de: $bids_root_dir"

# Detectar todos los sujetos tipo sub-XXXX
subjects=$(ls -d ${bids_root_dir}/sub-* 2>/dev/null | xargs -n1 basename)

if [ -z "$subjects" ]; then
    echo "❌ No se encontraron sujetos con prefijo 'sub-' en el directorio BIDS."
    exit 1
fi

echo "Sujetos encontrados:"
echo "$subjects"
echo ""

for subj in $subjects; do
    echo "==============================="
    echo " Procesando $subj"
    echo "==============================="

    outdir="${out_root}/${subj}"
    workdir="${outdir}/work"

    mkdir -p "$outdir"

    # Ejecutar MRIQC
    unset PYTHONPATH
    singularity run \
        --bind $bids_root_dir:/data:ro \
        --bind $outdir:/out \
        $container_path \
        /data /out \
        participant \
        --participant-label ${subj#sub-} \
        --n_proc $nthreads \
        --mem_gb $mem \
        --float32 \
        --ants-nthreads $nthreads \
        -w /out/work

    # ================================
    # BORRAR CARPETA WORK
    # ================================
    echo "Eliminando carpeta de trabajo temporal: $workdir"
    rm -rf "$workdir"

    echo "finished $subj"
    echo ""
done

echo "process finishesg"


