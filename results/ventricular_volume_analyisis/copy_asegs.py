import os
import shutil

# Ruta a la carpeta principal
ruta_principal = '/media/lautaro/SSDLauti/reconallCogConVar'
# Ruta de destino
ruta_destino = '/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/predictions/ventricular_volume_analyisis/asegs/asegsCogConVar'

# Crear la carpeta de destino si no existe
os.makedirs(ruta_destino, exist_ok=True)

# Recorrer todas las subcarpetas dentro de la carpeta principal
for sujeto_id in os.listdir(ruta_principal):
    ruta_sujeto = os.path.join(ruta_principal, sujeto_id)
    ruta_stats = os.path.join(ruta_sujeto, 'stats')
    archivo_aseg = os.path.join(ruta_stats, 'aseg.stats')

    if os.path.isfile(archivo_aseg):
        nuevo_nombre = f"{sujeto_id}aseg.stats"
        destino = os.path.join(ruta_destino, nuevo_nombre)
        shutil.copyfile(archivo_aseg, destino)
        print(f"Copiado: {archivo_aseg} → {destino}")
    else:
        print(f"Archivo no encontrado para: {sujeto_id}")
