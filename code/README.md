# Technical-validation code

These are the scripts used for the behavioral, QC/HRF, and inter-brain WTC analyses reported with the dataset.

## Environment

Python 3.11 or later is recommended. Install the Python dependencies with:

```powershell
python -m pip install -r code\requirements.txt
```

Homer3 v1.28.10 is required separately to generate the processed `.nirs` inputs for the QC/HRF and WTC scripts. The intermediate Homer3 files are not distributed. Recreate them from `sourcedata/` using `preprocessing_hrf_parameters.json` or `preprocessing_wtc_parameters.json`, respectively.

## Scripts

- `behavioral_validation.py`: condition summaries and within-dyad behavioral statistics.
- `qc_hrf_validation.py`: channel/trial quality control and channel-wise HbO/HbR HRF analysis.
- `wtc_interbrain_validation.py`: homologous-channel WTC, true-versus-pseudo dyad comparison with condition-wise paired tests, and WTC-behavior correlations. 

The scripts locate the dataset as the parent of this directory. Optional environment variables are:

- `COOPVM_DATASET_ROOT`: dataset root.
- `COOPVM_REPORT_ROOT`: output directory.
- `COOPVM_SOURCEDATA_ROOT`: native `.nirs` input for QC/HRF.
- `COOPVM_PROCESSED_ROOT`: Homer3 HRF/QC input directory.
- `COOPVM_PROCESSED_WTC_ROOT`: Homer3 WTC input directory.

Example:

```powershell
python code\behavioral_validation.py
$env:COOPVM_PROCESSED_ROOT = "D:\path\to\homer3-hrf"
python code\qc_hrf_validation.py
$env:COOPVM_PROCESSED_WTC_ROOT = "D:\path\to\homer3-wtc"
python code\wtc_interbrain_validation.py
```

The machine-readable outputs used in the data descriptor are provided under `derivatives/technical-validation/`.
