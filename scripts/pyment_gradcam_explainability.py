import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import sys
sys.path.append('/data/Lautaro/Documentos/BrainAgeCOVID/ADNI')
from pyment.models import RegressionSFCN
from tqdm import tqdm

IMAGE_FOLDER = '/data/Lautaro/Documentos/BrainAgeCOVID/DATOS/Preprocessed/pyment/prueba_explainability'
output_folder = os.path.join(os.getcwd(), 'gradcam_brainage_explainability')
os.makedirs(output_folder, exist_ok=True)

#cambiar device a GPU si está disponible
if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices([], 'CPU')  # Deshabilitar CPU si hay GPU disponible
    print("Usando GPU para Grad-CAM")
else:
    print("No se detectó GPU, usando CPU para Grad-CAM")

# Cargar modelo preentrenado
base_model = RegressionSFCN(weights='brain-age-2022')
base_model.trainable = False

# Obtener la última capa convolucional (antes del Global average pooling)
target_layer = base_model.get_layer('sfcn-reg_top_relu')
#grad model devuelve dos salidas: las activaciones de la capa objetivo y la prediccion
grad_model = Model(inputs=base_model.input, outputs=[target_layer.output, base_model.output])

for imageid in tqdm(os.listdir(IMAGE_FOLDER)):
    path = os.path.join(IMAGE_FOLDER, imageid, 'mri', 'cropped.nii.gz')
    subjectid = imageid[0:18]

    if not os.path.isfile(path):
        print(f'Skipping {imageid}: Missing cropped.nii.gz')
        continue

    img = nib.load(path).get_fdata()
    #se da la forma esperada por el modelo
    img = np.expand_dims(img, (0, -1))  # [1, H, W, D, 1]

    # Grad-CAM: obtener activaciones y calcular gradientes
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img) #conv_output es la activacion de la capa taret
        loss = prediction[0] #la prediccion esn si misma

    #se calcula el gradiente del loss respecto a la activacion
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))  # promedio global sobre espacio

    # Calcular el mapa Grad-CAM
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    #multiplica cada canal por su peso de importancia
    for i in range(pooled_grads.shape[-1]):
        conv_output[..., i] *= pooled_grads[i]

    #promedia sobre esos canales
    heatmap = np.mean(conv_output, axis=-1)
    #relu
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # normalización

    # Reescalar heatmap al tamaño original
    from scipy.ndimage import zoom
    print('img shape: ', img.shape)
    print('heatmap shape :',heatmap.shape)
    scale_factors = np.array(img.shape[1:4]) / np.array(heatmap.shape)
    print('scale factors: ', scale_factors)
    heatmap_resized = zoom(heatmap, zoom=scale_factors, order=1)
    print('heatmap resized shape: ', heatmap_resized.shape)


    #Mostrar y guardar overlay
    #se elige el voxel con la mayor activacion
    center = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    print('voxel mas representativo despues del rescalado: ', center)
    center_heatmap = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    print('voxel mas representativo antes del rescalado: ', center_heatmap)
    views = [
        ('Sagittal View', heatmap_resized[center[0], :, :], img[0, center[0], :, :, 0]),
        ('Coronal View', heatmap_resized[:, center[1], :], img[0, :, center[1], :, 0]),
        ('Axial View', heatmap_resized[:, :, center[2]], img[0, :, :, center[2], 0])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Grad-CAM - ID: {subjectid} - Predicción: {float(prediction[0]):.2f}")
    #se superpone la imagen original sobre el heatmap
    for ax, (title, heat, original) in zip(axes, views):
        heat = np.rot90(heat)
        original = np.rot90(original)
        ax.imshow(original, cmap='gray')
        ax.imshow(heat, cmap='hot', alpha=0.5)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{subjectid}_gradcam_overlay.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
