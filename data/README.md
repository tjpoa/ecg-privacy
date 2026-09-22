# Data layout

This project uses the [ECG Arrhythmia Database on
PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/).

The dataset must be obtained from PhysioNet and is not included in this
repository. Do not commit the downloaded waveforms, headers, or derived
patient-level tables.

After downloading the dataset, place the WFDB records locally as follows:

```text
data/
└── raw/
    └── WFDBRecords/
        ├── 01/
        ├── 02/
        └── ...
```

The code discovers header files recursively below
`data/raw/WFDBRecords/` and expects the corresponding `.mat` files beside each
`.hea` file.

The utility task maps SNOMED-CT codes `164889003` (atrial fibrillation) and
`164890007` (atrial flutter) to the positive atrial-arrhythmia class. The
dataset's own documentation and citation must be followed when the data are
used or redistributed.

Generated intermediate and processed files are written under `data/interim/`
and `data/processed/`; both locations are ignored by Git.
