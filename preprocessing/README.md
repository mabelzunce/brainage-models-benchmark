# Preprocessing Scripts

This directory contains the preprocessing scripts used for the raw T1-weighted neuroimages in the dataset.

## Pipelines

- **BrainAgeNeXt**  
  • N4 bias field correction  
  • Skullstripping using SynthStrip  
  • Rigid registration into MNI space with ANTs

- **DeepBrainNet**  
  • N4 bias field correction  
  • Skullstripping using BET and SynthStrip  
  • Affine registration into MNI space with ANTs

- **Pyment**  
  • Skullstripping using Freesurfer's recon-all pipeline  
  • Reorientation with fslreorient2std  
  • Rigid registration into MNI space  
  • Image cropping

- **ENIGMA**  
  • Freesurfer's recon-all pipeline for regional thicknesses and volumes

## Quality Check

The `quality_check` directory contains 2D images for every subject from each model and dataset combination. Each image shows the before and after stages of the preprocessing pipelines.