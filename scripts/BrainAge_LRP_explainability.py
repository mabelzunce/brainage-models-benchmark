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
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
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

# -------------------------
# LRP con Zennit
# -------------------------
def run_predictions_with_lrp(model_path, dataloader, save_individual_nifti=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedNeXtEncReg().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Composite disponible en tu Zennit
    composite = EpsilonPlus(epsilon=1e-6)
    # Attributor de Zennit (maneja el context de composite internamente)
    attributor = Gradient(model, composite=composite)

    output_folder = os.path.join(os.getcwd(), 'LRP_brainage_explainability')
    os.makedirs(output_folder, exist_ok=True)

    predictions = []

    for idx, batch in enumerate(dataloader):
        img = batch['image'].to(device)  # [1,1,D,H,W]

        # Ejecuta forward + atribución en un solo paso.
        # attr_output=lambda y: y.sum() garantiza un escalar (útil con regresión [B,1]).
        output, relevance = attributor(img, attr_output=lambda y: y)
        # Guardá la predicción (numérico) por conveniencia:
        predictions.append(float(output.detach().cpu().numpy().ravel()[0]))

        # Relevancia con misma forma que la entrada
        rel = relevance.detach().cpu().numpy().squeeze()  # [D,H,W]

        # NIfTI de referencia
        filename = batch['image'].meta.get('filename_or_obj')
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        reference_nii = nib.load(filename)

        # Reescalar relevancia al espacio original
        target_shape = np.array(reference_nii.shape[:3])
        scale = target_shape / np.array(rel.shape)
        rel_resized = scipy.ndimage.zoom(rel, zoom=scale, order=1)

        # Guardar NIfTI individual si querés
        if save_individual_nifti:
            out_path = os.path.join(output_folder, f"lrp_subject_{idx+1}.nii.gz")
            nib.save(nib.Nifti1Image(rel_resized.astype(np.float32), affine=reference_nii.affine), out_path)

        # (Opcional) también podés generar tu overlay PNG como ya hacías

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
    df.to_csv(csv_file.replace('.csv', '_with_lrp.csv'), index=False)
    print("Resultados con LRP individuales guardados.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <path_csv>")
        sys.exit(1)
    main(sys.argv[1])

