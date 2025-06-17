import os
import pandas as pd

# Ruta del directorio a recorrer
directory = "/data/Lautaro/Documentos/BrainAgeCOVID/ADNI/ADNI_3_T1/brainagenext/preprocesedCP"
files = []

# Recorrer recursivamente el directorio para obtener todos los archivos
for root, _, filenames in os.walk(directory):
    for filename in filenames:
        files.append(os.path.join(root, filename))

# Crear un DataFrame con la columna "path"
df = pd.DataFrame({'path': files})
# Guardar el DataFrame en un CSV
csv_out_path = os.path.join(directory, 'csv_file.csv')
df.to_csv(csv_out_path, index=False)

print(f"CSV generado en: {csv_out_path}")