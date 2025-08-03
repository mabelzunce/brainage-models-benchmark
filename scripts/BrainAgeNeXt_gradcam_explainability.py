import os
import torch
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
from monai.data import CacheDataset
import torch.nn as nn
import torch.nn.functional as F
import torchio
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from nnunet_mednext import create_mednext_encoder_v1
from monai.transforms import NormalizeIntensityd
import scipy.ndimage

#defino la clase del modelo
class MedNeXtEncReg(nn.Module):
    def __init__(self):
        super(MedNeXtEncReg, self).__init__()
        #un canal de entrada, una salida, model_id elige el tamaño del modelo, filtro de tamaño 3
        self.mednextv1 = create_mednext_encoder_v1(
            num_input_channels=1, num_classes=1, model_id='B', kernel_size=3, deep_supervision=True
        )
        #inicializo los atributos donde voy a guardar los gradientes y las activaciones
        self.target_activations = None
        self.target_gradients = None
        #convierto el volumen 3d a un unico valor por canal
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #regresion lineal sobre los features
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, 1)
        )
        # Hook para Grad-CAM
        target_layer = self.mednextv1.bottleneck[1]
        target_layer.register_forward_hook(self.save_activation) #guarda las activaciones cuando pasa info hacia adelante
        target_layer.register_full_backward_hook(self.save_gradient) #guarda los gradientes cuando hace backpropagation

    def save_activation(self, module, input, output):
        self.target_activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.target_gradients = grad_output[0]

    def forward(self, x):
        x = self.mednextv1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.regression_fc(x)

def compute_gradcam(activations, gradients):
    #calcula el promedio de los gradientes para cada canal, estos pesos tienen la importancia que tiene cada canal para la prediccion:
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True) 
    #multiplica las activaciones por los pesos y los suma para tener un mapa de activacion ponderado
    cam = (weights * activations).sum(dim=1)
    #aplico una capa relu para quedarme solo con las contribuciones positivas
    #cam = F.relu(cam)
    #normalizo el mapa para quedarme con valores entre 0 y 1
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam

def prepare_transforms():
    #transformaciones que le realizo a las imagenes antes de meterlas
    x, y, z = (160, 192, 160) #volumen destino
    p = 1.0 #resolucion (1mm)
    monai_transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    ]
    #guardo las transformaciones como un compose (para que sea una transformacion que incluye muchas)
    return Compose(monai_transforms)

def load_data(csv_file):
    #lee el csv
    df = pd.read_csv(csv_file).dropna(subset=['Path', 'Age'])
    #data_dicts es un diccionario con la edad y el path de cada nifti
    data_dicts = [{'image': row['Path'], 'label': row['Age']} for _, row in df.iterrows()]
    return df, data_dicts

#creo el dataloader:
def create_dataloader(data_dicts, transforms):
    #creo el dataset y les aplico las transformaciones
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=4)
    #creo el dataloader
    return DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())

def run_predictions_with_gradcam(model_path, dataloader):
    print('entro a run_predictions with gradcam:')
    #instancio el modelo y lo pongo en modo evaluacion:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedNeXtEncReg().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    output_folder = os.path.join(os.getcwd(), 'brainage_explainability')
    os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
    predictions = []
    for batch_data in dataloader:
        images = batch_data['image'].to(device)
        images.requires_grad_(True)
        #hago las predicciones
        pred = model(images)  # shape: [1, 1]
        predictions.append(pred.item())

        #limpio los gradientes anteriores
        model.zero_grad()
        #hago backpropagation del gradiente desde la prediccion hacia la capa objetivo
        pred.mean().backward()

        #obtengo el mapa de gradcam
        cam = compute_gradcam(
            #usa las activaciones y los gradientes guardados por los hooks
            model.target_activations.detach(),
            model.target_gradients.detach()
        ).squeeze().cpu().numpy()

        #obtengo la imagen de referencia
        image_meta = batch_data["image"].meta
        ref_img_path = image_meta["filename_or_obj"]
        ref_img = nib.load(ref_img_path)

        print('cam shape ', cam.shape)
        print('reference image shape: ', ref_img.shape )
        max_coords_cam = np.unravel_index(np.argmax(cam), cam.shape)
        print(f"Coordenadas del voxel más representativo antes del reshape: {max_coords_cam}")

        # Obtener las formas originales
        target_shape = ref_img.shape
        cam_array = cam
        print('cam_array shape: ', cam_array.shape)
        # Calcular los factores de escalado
        scale_factors = np.array(target_shape) / np.array(cam_array.shape)

        print('factor de escalado', scale_factors)
        # Interpolar el Grad-CAM al tamaño de la imagen original
        cam_resized = scipy.ndimage.zoom(cam_array, zoom=scale_factors, order=1)  # orden 1: bilineal  (osea bordes difuminados)
        print('cam resized shape: ',cam_resized.shape)
        # Crear un nuevo NIfTI con el espacio ajustado
        cam_resized_nii = nib.Nifti1Image(cam_resized, affine=ref_img.affine)

        # Guardar el archivo NIfTI en la carpeta "brainage_explainability"
        subj_id = os.path.basename(ref_img.get_filename()).replace('.nii.gz', '')
        output_path = os.path.join(output_folder, f'gradcam_{subj_id}.nii.gz')
        #nib.save(cam_resized_nii, output_path)
        print(f"Grad-CAM guardado en: {output_path}")

        # Guardar mapa como overlay PNG
        overlay_filename = f"{subj_id}_overlay.png"
        overlay_path = os.path.join(output_folder, overlay_filename)
        save_overlay_image(ref_img, cam_resized_nii, overlay_path)
    print('salgo de run predictions with gradcam')
    return np.array(predictions)

def save_overlay_image(image_nii, cam_nii, output_path, alpha=0.4, slice_axis=2):
    print('entro a save overlay image')
    img_data = image_nii.get_fdata()
    cam_data = cam_nii.get_fdata()
    print('\n-------------------------------------------')
    print('cam_data shape:', cam_data.shape)
    print('img_data shape:', img_data.shape)
    print('-------------------------------------------') 

    # Normalizar imagen y gradcam
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    cam_data = (cam_data - cam_data.min()) / (cam_data.max() - cam_data.min() + 1e-8)

    max_coords = np.unravel_index(np.argmax(cam_data), cam_data.shape)
    print(f"Coordenadas del voxel más representativo: {max_coords}")

    # Obtener los cortes en las coordenadas del voxel más representativo
    img_slice_axial = np.rot90(img_data[:, :, max_coords[2]])
    cam_slice_axial = np.rot90(cam_data[:, :, max_coords[2]])

    img_slice_coronal = np.rot90(img_data[:, max_coords[1], :])
    cam_slice_coronal = np.rot90(cam_data[:, max_coords[1], :])

    img_slice_sagittal = np.rot90(img_data[max_coords[0], :, :])
    cam_slice_sagittal = np.rot90(cam_data[max_coords[0], :, :])

    # Crear el gráfico con tres subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #modificar el titulo para que aparezca el ID del sujeto
    fig.suptitle(f"Grad-CAM most relevant voxels, ID: {os.path.basename(image_nii.get_filename()).replace('.nii.gz', '')})", fontsize=16)

    # Sagittal view
    axes[0].imshow(img_slice_sagittal, cmap='gray')
    axes[0].imshow(cam_slice_sagittal, cmap='hot', alpha=alpha)
    axes[0].set_title("Sagittal View")
    axes[0].axis('off')

    # Coronal view
    axes[1].imshow(img_slice_coronal, cmap='gray')
    axes[1].imshow(cam_slice_coronal, cmap='hot', alpha=alpha)
    axes[1].set_title("Coronal View")
    axes[1].axis('off')

    # Axial view
    axes[2].imshow(img_slice_axial, cmap='gray')
    axes[2].imshow(cam_slice_axial, cmap='hot', alpha=alpha)
    axes[2].set_title("Axial View")
    axes[2].axis('off')

    # Ajustar el diseño y guardar el gráfico
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print('salgo de save overlay image')


def main(csv_file):
    #cargo el csv
    df, data_dicts = load_data(csv_file)
    
    #defino las transformaciones
    transforms = prepare_transforms()
    
    #instancio el dataloader
    dataloader = create_dataloader(data_dicts, transforms)
    #path del modelo
    model_path = os.path.join(os.path.dirname(__file__), 'BrainAge_4.pth')

    #corro las predicciones con gradcam
    predictions = run_predictions_with_gradcam(model_path, dataloader)

    CA = df['Age'].values
    BA = predictions
    #para los sujetos mayores a 18 años hago una correccion de la edad para evitar regression-to-the-mean bias
    BA_corr = np.where(CA > 18, BA + (CA * 0.062) - 2.96, BA)
    BAD_corr = BA_corr - CA

    df['Predicted_Brain_Age'] = BA_corr
    df['Brain_Age_Difference'] = BAD_corr
    df.to_csv(csv_file.replace('.csv', '_with_predictions.csv'), index=False)
    print('Updated CSV file and Grad-CAM maps saved.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: No .csv file provided.")
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    main(sys.argv[1])