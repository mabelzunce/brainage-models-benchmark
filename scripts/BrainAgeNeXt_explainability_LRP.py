import sys
import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, Spacingd, CropForegroundd,
    SpatialPadd, CenterSpatialCropd, NormalizeIntensityd
)
from monai.data import CacheDataset
import torch.nn as nn
import scipy.ndimage
from nilearn.image import resample_to_img

# Asegurate de añadir tu path al nnunet_mednext
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from nnunet_mednext import create_mednext_encoder_v1

# Zennit imports
from zennit.composites import Composite
from zennit.rules import Epsilon, Pass
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus

from matplotlib.colors import TwoSlopeNorm

# -------------------------
# Modelo
# -------------------------
class MedNeXtEncReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.mednextv1 = create_mednext_encoder_v1(
            num_input_channels=1, num_classes=1,
            model_id='B', kernel_size=3, deep_supervision=True
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.mednextv1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.regression_fc(x)

# -------------------------
# Transforms y dataloader
# -------------------------
def prepare_transforms():
    x, y, z = (160, 192, 160)
    p = 1.0
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        #remuestrea la imagen a 1mm isotrópico
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        #se elimina el fondo (valores de 0)
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        #si la imagen es más chica que el tamaño objetivo, se rellena con ceros
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        #se vuelve a recortar la imagen desde el centro al tamaño objetivo
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])

def load_data(csv_file):
    df = pd.read_csv(csv_file).dropna(subset=['Path', 'Age'])
    data_dicts = [{'image': r['Path'], 'label': r['Age']} for _, r in df.iterrows()]
    return df, data_dicts

def create_dataloader(data_dicts, transforms):
    ds = CacheDataset(data=data_dicts, transform=transforms,
                      cache_rate=0.2, num_workers=4)
    return DataLoader(ds, batch_size=1, num_workers=4,
                      shuffle=False, pin_memory=torch.cuda.is_available())

from matplotlib.colors import TwoSlopeNorm

def save_relevance_png_3views(rel_data, out_path, perc_clip=99):
    # Clipping simétrico para mejorar contraste
    vmax = np.percentile(np.abs(rel_data), perc_clip)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Voxel más representativo
    idx_max = np.unravel_index(np.argmax(np.abs(rel_data)), rel_data.shape)

    # Slices rotados 180°
    slices = [
        np.fliplr(np.rot90(rel_data[idx_max[0], :, :], k=3)),  # Sagittal
        np.rot90(rel_data[:, idx_max[1], :], k=3),  # Coronal
        np.rot90(rel_data[:, :, idx_max[2]], k=3)   # Axial
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    views = ['Sagittal View', 'Coronal View', 'Axial View']

    cmap = plt.cm.seismic
    im = None

    for ax, slc, view in zip(axes[:3], slices, views):
        im = ax.imshow(slc, cmap=cmap, norm=norm, origin='lower')
        ax.set_title(view, fontsize=16)
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_aspect('equal')

    fig.suptitle('Average LRP Relevance Map UNSAM_LC', fontsize=20,x=0.45)
    cbar = plt.colorbar(im, cax=axes[3])
    cbar.set_label('Relevance', rotation=270, labelpad=15,fontsize=16)

    # Ajustar sin deformar
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    

# -------------------------
# LRP con Zennit
# -------------------------
def run_predictions_with_lrp(model_path, dataloader, save_individual_nifti=True, save_individual_png=True, save_mean_png=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedNeXtEncReg().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    composite = EpsilonPlus(epsilon=1e-6)
    attributor = Gradient(model, composite=composite)

    output_folder = os.path.join(os.getcwd(), 'LRP_brainage_explainability')
    os.makedirs(output_folder, exist_ok=True)

    predictions = []
    mean_relevance_sum = None
    reference_nii = None
    count = 0

    for idx, batch in enumerate(dataloader):
        img = batch['image'].to(device)  # [1,1,D,H,W]
        #me quedo con el affine de la primera imagen
        
        output, relevance = attributor(img, attr_output=lambda y: y)
        predictions.append(float(output.detach().cpu().numpy().ravel()[0]))

        rel = relevance.detach().cpu().numpy().squeeze()

        filename = batch['image'].meta.get('filename_or_obj')
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        
        affine_curr = batch['image'].meta.get("affine", np.eye(4))

        # si es la primera vez, cargo la referencia como NIfTI real
        if reference_nii is None:
            reference_nii = nib.load(filename)
       
        rel_nii = nib.Nifti1Image(rel.astype(np.float32), affine_curr)

        # resampleo al espacio de la referencia
        rel_resampled = resample_to_img(rel_nii, reference_nii, interpolation="nearest").get_fdata()
        rel_resampled = rel_resampled / np.amax(np.abs(rel_resampled))

        if mean_relevance_sum is None:
            mean_relevance_sum = np.zeros_like(rel_resampled)
        mean_relevance_sum += rel_resampled
        count += 1

        print(count)
        #if save_individual_png:
            #out_path = os.path.join(output_folder, f"lrp_subject_{idx+1}.png")
            #save_overlay_png_3views(reference_nii, rel_resized, out_path)

    # Guardar PNG del promedio final
    if save_mean_png and count > 0:
        mean_relevance = mean_relevance_sum / count
        out_path = os.path.join(output_folder, "lrp_mean.png")
        save_relevance_png_3views(mean_relevance, out_path)
        # Guardar NIfTI del promedio
        out_nii_mean = nib.Nifti1Image(mean_relevance.astype(np.float32), reference_nii.affine)
        #out_path_nii_mean = os.path.join(output_folder, "lrp_mean_CogConVar.nii.gz")
        #nib.save(out_nii_mean, out_path_nii_mean)

    return np.array(predictions)


# -------------------------
# Main
# -------------------------
def main(csv_file):
    df, dicts = load_data(csv_file)
    dl = create_dataloader(dicts, prepare_transforms())
    preds = run_predictions_with_lrp(
        os.path.join(os.path.dirname(__file__), 'BrainAge_1.pth'),
        dl
    )

    CA = df['Age'].values
    BA = preds
    BA_corr = np.where(CA > 18, BA + CA*0.062 - 2.96, BA)
    df['Predicted_Brain_Age'] = BA_corr
    df['Brain_Age_Difference'] = BA_corr - CA
    #df.to_csv(csv_file.replace('.csv', '_with_lrp.csv'), index=False)
    print("Resultados con LRP individuales guardados.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <path_csv>")
        sys.exit(1)
    main(sys.argv[1])

