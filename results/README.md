# Analysis of Brain Age Prediction Results

This directory contains the results and analysis scripts for brain age prediction models. Each model folder includes predictions for every subject from the tested datasets.

## Contents

- **Model Predictions**  
  Folders for **BrainAgeNeXt**, **DeepBrainNet**, **ENIGMA**, and **Pyment** contain brain age predictions for every subject from the datasets: ADNI, UNSAM_LC, JUK, and NCN.

- **Ventricular Volume Analysis**  
  Contains scripts and CSV files used to upload each subject's ventricular size, cortical/subcortical grey matter volume, and white matter volume into the brainchart web application to obtain centile scores relative to normative trajectories.

## Notebooks

- **scatter+box.ipynb**  
  Calculates MAE and ME for every model-dataset combination, generates scatterplots of real vs predicted age, and produces Brain Age Gap boxplots for every model and clinical subgroup.

- **meanMAE.ipynb**  
  Computes overall MAE, ME, STD, and ASTD for every model using a random sample of 97 subjects from each dataset (totaling 388 subjects).

- **demohists.ipynb**  
  Generates demographic histograms for every dataset.

- **bland-altman.ipynb**  
  Creates Bland-Altman plots for the predictions.
