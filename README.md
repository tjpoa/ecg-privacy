# ECG Privacy Risk Experiment

Code and reproducibility material for Study 2 (ECG) of the thesis and the
associated journal manuscript. The project evaluates privacy risk in released
ECG representations while measuring atrial-arrhythmia utility on the same
representations. The exact submission protocol is recorded in
[`configs/experiment_config.yaml`](configs/experiment_config.yaml) and
[`docs/final_protocol.md`](docs/final_protocol.md).

The repository is organised for public code availability. Raw ECG files,
locally generated processed datasets, fitted models, and exploratory handoff
documents are intentionally excluded from the public release.

## What is evaluated

- ECG preprocessing, quality checks, segmentation, and handcrafted features;
- atrial-arrhythmia utility classification;
- segment linkability and operational-style gallery linkage;
- staged attribute-inference attacks before and after attacker-selected segment
  aggregation;
- empirical 12-lead waveform-reconstruction attacks;
- generic projections, feature-selection strategies, and a supervised
  bottleneck representation;
- calibrated perturbation of released embeddings and the resulting
  privacy--utility trade-off.

The calibrated embedding perturbation is not DP-SGD and should not be
interpreted as end-to-end differentially private training.

## Data

The experiments use the [ECG Arrhythmia Database on
PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/). The data are
not redistributed by this repository. Follow the dataset's access, citation,
and licence requirements; see [data/README.md](data/README.md) for the expected
local layout.

The source dataset is publicly available under its stated CC BY 4.0 licence.
The repository stores only code and metadata needed to describe the expected
inputs, not ECG waveforms or derived patient-level datasets.

## Installation

Python 3.10.13 is the recorded analysis environment. The package versions are
pinned in `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick checks

Run the release audit and unit tests during preparation:

```powershell
python scripts\verify_public_release.py
python -m unittest discover -s tests -v
```

Immediately before publication, require the selected licence as well:

```powershell
python scripts\verify_public_release.py --require-license
```

The release audit checks for raw data, temporary directories, local absolute
paths, manuscript and PDF artefacts, common secret patterns, inconsistent CSV
rows, and the minimum documentation files required for reuse.

## Reproducing the pipeline

After placing the PhysioNet files under `data/raw/WFDBRecords/`, build the
feature dataset in batches. The following is a small smoke run:

```powershell
python src\build_segment_features_dataset.py --record-limit 1000 --batch-size 250 --window-sec 2.0 --step-sec 1.0
```

The full command sequence and expected artefacts are documented in
[`docs/reproduction.md`](docs/reproduction.md).

The two attack extensions can be reproduced with the final encoder protocol:

```powershell
python src\run_attribute_inference_attack.py --max-chunks 0 --seeds 42 123 456 --encoder-max-iter 25 --encoder-train-max-segments 40000 --run-name attribute_inference_final_protocol_multiseed
python src\run_reconstruction_attack.py --max-chunks 0 --seeds 42 123 456 --target-records 12000 --target-pca-components 128 --encoder-max-iter 25 --encoder-train-max-segments 40000 --run-name reconstruction_final_protocol_multiseed
python src\summarize_attack_extensions.py --attribute-prefix outputs\tables\attribute_inference_final_protocol_multiseed --reconstruction-prefix outputs\tables\reconstruction_final_protocol_multiseed --run-name privacy_attack_extensions_final_protocol
```

The reconstruction runner caches locally derived waveform targets under
`data/interim/`. These targets and all per-record predictions remain excluded
from the public repository.

## Repository layout

```text
configs/       experiment configuration
data/          local data instructions and ignored data directories
docs/          final protocol and reproduction instructions
figures/       reproducible public PNG figures
reproducibility/ aggregate tables supporting the manuscript
scripts/       release checks and small utility runners
src/           reusable preprocessing, modelling, and privacy code
tests/         lightweight regression tests
```

## Reproducibility conventions

- Splits are performed at the patient/record level to avoid segment leakage.
- Final robustness analyses use explicit random seeds, including 42, 123, and
  456 where indicated by the protocol.
- Full protocol outputs and per-record results are generated locally and remain
  ignored. A curated set of aggregate article tables is stored under
  `reproducibility/`.
- Claims about privacy are protocol-specific empirical findings, not a proof of
  anonymity or a universal privacy guarantee.
- The Gaussian operating points are reported by clipping norm, noise standard
  deviation, and noise multiplier. They do not provide an end-to-end
  differential-privacy guarantee.

## Code and data availability

The source ECG recordings are available from the [PhysioNet ECG Arrhythmia
Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/), subject to the
dataset's access, citation, and licensing terms. The recordings, headers,
patient-level data, and derived per-record files are not redistributed in this
repository.

The author-owned source code, experiment configurations, documentation, tests,
public figures, and aggregate result tables are available at
<https://github.com/tjpoa/ecg-privacy>. The code is released under the
[MIT License](LICENSE).

Full reproduction instructions are provided in
[`docs/reproduction.md`](docs/reproduction.md). Before submission, the
manuscript should cite a tagged release or Zenodo DOI for the exact code
version used.

## Licence

The author-owned code in this repository is released under the [MIT License](LICENSE).
This licence does not relicense third-party material; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The final author list and
institutional ownership should still be confirmed before creating an official
release.
