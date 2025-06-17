import os
import shutil

jpg_dir = "/home/lautaro/Documentos/BrainAgeNeXt/MedNeXt/Preprocessing_analysis"
nii_dir = "/data/Lautaro/Documentos/BrainAgeCOVID/ADNI/ADNI_3_T1/brainagenext/Archivos_PreprocesadosIXI"
output_dir = "/data/Lautaro/Documentos/BrainAgeCOVID/ADNI/ADNI_3_T1/brainagenext/filtered_niisIXI"

# Obtener IDs desde los nombres .jpg (sin extensión)
jpg_files = os.listdir(jpg_dir)
ids = [os.path.splitext(f)[0] for f in jpg_files if f.endswith(".jpg")]

# Copiar solo los .nii.gz que tienen imagen .jpg correspondiente
for id_name in ids:
    src = os.path.join(nii_dir, f"{id_name}.nii.gz")
    dst = os.path.join(output_dir, f"{id_name}.nii.gz")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Copiado: {id_name}.nii.gz")
    else:
        print(f"No encontrado: {id_name}.nii.gz")


