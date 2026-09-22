# Reproduction instructions

These commands reproduce the article workflow after the PhysioNet records have
been placed under `data/raw/WFDBRecords/`. They are intended to be run from the
repository root on Windows PowerShell. Full runs are computationally expensive.

## 1. Create the environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Build the final segment dataset

```powershell
python src\build_segment_features_dataset.py --batch-size 250 --window-sec 2.0 --step-sec 1.0
```

The expected result is 45,151 retained records, nine segments per record, and
406,359 rows under `data/processed/final_segment_features_dataset/`.

## 3. Run the matched representation comparison

```powershell
python src\run_representation_dimension_comparison.py --max-chunks 0 --seeds 42 123 456 --run-name representation_dimension_submission_matched --methods identity pca random_projection supervised_encoder --method-dims pca:8 pca:64 random_projection:8 random_projection:64 supervised_encoder:8 supervised_encoder:64 --encoder-max-iter 25 --encoder-train-max-segments 40000 --shared-train-max-segments 40000 --link-max-pairs 2000
python src\summarize_record_level_utility.py --predictions outputs\tables\representation_dimension_submission_matched_utility_predictions.csv --run-name representation_dimension_submission_matched_record_level
python src\run_representation_dimension_comparison.py --max-chunks 0 --seeds 42 123 456 --run-name article_controlled_representation_multiseed --methods identity pca random_projection supervised_encoder encoder_gaussian_c2 encoder_gaussian_c4 --method-dims pca:64 random_projection:64 supervised_encoder:8 encoder_gaussian_c2:8 encoder_gaussian_c4:8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --shared-train-max-segments 40000 --link-max-pairs 2000
python src\summarize_record_level_utility.py --predictions outputs\tables\article_controlled_representation_multiseed_utility_predictions.csv --run-name article_controlled_record_level_utility
```

Generate the final paired summaries and figure with:

```powershell
python src\export_main_paired_uncertainty.py
python src\generate_main_tradeoff_figure.py
```

## 4. Run the operational gallery

```powershell
python src\run_operational_linkage_eval.py --max-chunks 0 --seeds 42 123 456 --gallery-negatives 99 999 --queries-per-seed 1000 --run-name operational_linkage_final
python src\run_encoder_dp_operational_linkage_eval.py --max-chunks 0 --seeds 42 123 456 --run-name encoder_dp_operational_dim8_multiseed --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --train-max-pairs 2000 --gallery-negatives 99 999 --queries-per-seed 1000 --dp-configs 50:2 50:4
```

The `epsilon:clip` command syntax is retained by the implementation for
historical configuration compatibility. The article reports the realised
clipping norm, noise standard deviation, and noise multiplier and does not
claim an end-to-end differential-privacy guarantee.

## 5. Run attribute and reconstruction attacks

```powershell
python src\run_attribute_inference_attack.py --max-chunks 0 --seeds 42 123 456 --encoder-max-iter 25 --encoder-train-max-segments 40000 --run-name attribute_inference_final_protocol_multiseed
python src\run_reconstruction_attack.py --max-chunks 0 --seeds 42 123 456 --target-records 12000 --target-pca-components 128 --encoder-max-iter 25 --encoder-train-max-segments 40000 --run-name reconstruction_final_protocol_multiseed
python src\run_conditioned_reconstruction_attack.py --max-chunks 0 --seeds 42 123 456 --target-records 12000 --target-pca-components 128 --encoder-max-iter 25 --encoder-train-max-segments 40000 --run-name conditioned_reconstruction_final_protocol_multiseed
python src\summarize_attack_extensions.py --attribute-prefix outputs\tables\attribute_inference_final_protocol_multiseed --reconstruction-prefix outputs\tables\reconstruction_final_protocol_multiseed --run-name privacy_attack_extensions_final_protocol
python src\export_absolute_attribute_risk.py
```

## 6. Validate the release candidate

```powershell
python scripts\verify_public_release.py --require-license
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

Generated data, predictions, models, and per-record files remain under ignored
local directories. The public repository includes only the final PNG figures
and aggregate tables needed to interpret and reproduce the article workflow.
