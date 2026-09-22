# Notebook Guide

This project has been consolidated into a smaller set of main notebooks.

## Main workflow

1. `01_data_understanding.ipynb`
   - dataset validation, exploratory analysis, and initial observations.

2. `02_data_preparation.ipynb`
   - raw record loading, preprocessing, segmentation, feature extraction, and final processed dataset summary.

3. `03_modeling_and_tradeoff.ipynb`
   - segmentation selection, final baselines, utility feature selection, utility/linkability feature-importance overlap, same-data trade-off, progressive feature removal, and record-level utility.

4. `04_privacy_analysis.ipynb`
   - linkability robustness, global privacy transformations, selective privacy transformations, PCA component analysis, and sensitivity to temporal gap / hard negatives.
   - final split-aware transformation runs can be reproduced with `src/run_privacy_transform_study.py`.
   - selective feature-guided transformation runs can be reproduced with `src/run_selective_privacy_transform_study.py`.
   - supervised bottleneck encoder runs can be reproduced with `src/run_supervised_bottleneck_encoder_study.py`.
   - encoder operational-style linkage runs can be reproduced with `src/run_encoder_operational_linkage_eval.py`.
   - aggregate-view linkability runs can be reproduced with `src/run_aggregate_linkability_study.py`.
   - operational-style linkage metrics can be reproduced with `src/run_operational_linkage_eval.py`.

5. `05_results_summary.ipynb`
   - compact synthesis of the main experimental findings.

## Archived notebooks

Detailed and more specialized notebooks from the exploratory/development phase are stored in `notebooks/archive/`.
They are kept for traceability, but the recommended reading order for the thesis workflow is the five notebooks above.

## Windowing selection evidence

The Study II windowing choice is documented as a pragmatic operating point rather than an optimized/Pareto-selected configuration.
The recovered quantitative evidence is stored in:

- `outputs/tables/study2_windowing_selection_evidence.csv`
- `outputs/tables/baseline_sweep_summary.csv`
- `notebooks/archive/05_segmentation_selection.ipynb`

The first file consolidates the `1000`-record four-configuration pilot and the `5000`-record focused comparison between `w2_o0p5` and `w3_o0p5`.

## Computational Protocol

The model/environment protocol tables can be regenerated with:

```powershell
.\venv\Scripts\python.exe src\export_model_protocol.py
```

The canonical artefacts are:

- `outputs/tables/computational_environment_versions.csv`
- `outputs/tables/model_computational_protocol_summary.csv`
- `outputs/tables/model_hyperparameters.csv`

## Utility/Linkability Feature-Importance Overlap

Use this command for the full-dataset comparison of which feature families support utility versus linkability:

```powershell
.\venv\Scripts\python.exe src\run_feature_group_importance_analysis.py --max-chunks 0 --run-name feature_group_importance_full
```

The canonical artefacts are:

- `outputs/tables/feature_group_importance_full_feature_importance_long.csv`
- `outputs/tables/feature_group_importance_full_group_summary.csv`
- `outputs/tables/feature_group_importance_full_feature_overlap.csv`
- `outputs/tables/feature_group_importance_full_customization_candidates.csv`
- `outputs/tables/feature_group_importance_full_protocol.json`
- `outputs/figures/feature_importance_stat_groups.png`
- `outputs/figures/feature_importance_lead_groups.png`
- `outputs/figures/feature_importance_overlap_scatter.png`

Use `--max-chunks 20 --run-name feature_group_importance_20chunks` for a faster exploratory smoke run.

## Selective privacy transformations

Use this command to reproduce the current seed-42 selective PCA check using the top-80 utility/linkability feature-overlap candidates:

```powershell
.\venv\Scripts\python.exe src\run_selective_privacy_transform_study.py --max-chunks 0 --seeds 42 --run-name selective_privacy_transform_top80_seed42 --candidates-path .\outputs\tables\feature_group_importance_full_top80_customization_candidates.csv --top-per-type 80 --pca-components 10 20 30 --link-max-pairs 2000 --experiments identity link_pca_10 link_pca_20 link_pca_30
```

The current artefacts are:

- `outputs/tables/selective_privacy_transform_top80_seed42_metrics.csv`
- `outputs/tables/selective_privacy_transform_top80_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_top80_noise_seed42_metrics.csv`
- `outputs/tables/selective_privacy_transform_top80_noise_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_top80_combined_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_preserve_utility_rest_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_preserve_utility50_rest_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_preserve_utility30_rest_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_all_seed42_summary.csv`
- `outputs/tables/selective_privacy_transform_top80_seed42_feature_policy.csv`
- `outputs/tables/selective_privacy_transform_top80_seed42_protocol.json`
- `outputs/figures/selective_privacy_transform_top80_tradeoff.png`
- `outputs/figures/selective_privacy_transform_top80_combined_tradeoff.png`
- `outputs/figures/selective_privacy_transform_all_seed42_frontier.png`

This run is useful as directional evidence for customized privacy, but it is still a single-seed exploratory check.

The selective noise/winsorization variants can be reproduced with:

```powershell
.\venv\Scripts\python.exe src\run_selective_privacy_transform_study.py --max-chunks 0 --seeds 42 --run-name selective_privacy_transform_top80_noise_seed42 --candidates-path .\outputs\tables\feature_group_importance_full_top80_customization_candidates.csv --top-per-type 80 --link-max-pairs 2000 --experiments link_noise_005 link_noise_010 link_noise_020 tiered_noise_l010_s003 tiered_noise_l020_s005 link_winsor05_noise010 tiered_winsor05_noise_l010_s003
```

The more aggressive preserve-utility/rest-protected variants can be reproduced with:

```powershell
.\venv\Scripts\python.exe src\run_selective_privacy_transform_study.py --max-chunks 0 --seeds 42 --run-name selective_privacy_transform_preserve_utility_rest_seed42 --candidates-path .\outputs\tables\feature_group_importance_full_top80_customization_candidates.csv --top-per-type 80 --pca-components 20 40 --link-max-pairs 2000 --experiments identity preserve_utility_pca_rest_20 preserve_utility_pca_rest_40 preserve_utility_rp_rest_40

.\venv\Scripts\python.exe src\run_selective_privacy_transform_study.py --max-chunks 0 --seeds 42 --run-name selective_privacy_transform_preserve_utility50_rest_seed42 --candidates-path .\outputs\tables\feature_group_importance_full_top80_customization_candidates.csv --top-per-type 50 --pca-components 40 --link-max-pairs 2000 --experiments preserve_utility_pca_rest_40 preserve_utility_rp_rest_40

.\venv\Scripts\python.exe src\run_selective_privacy_transform_study.py --max-chunks 0 --seeds 42 --run-name selective_privacy_transform_preserve_utility30_rest_seed42 --candidates-path .\outputs\tables\feature_group_importance_full_top80_customization_candidates.csv --top-per-type 30 --pca-components 40 --link-max-pairs 2000 --experiments preserve_utility_rp_rest_40
```

## Supervised bottleneck encoder

Use these commands to reproduce the current supervised bottleneck encoder results:

```powershell
.\venv\Scripts\python.exe src\run_supervised_bottleneck_encoder_study.py --max-chunks 0 --seeds 42 --run-name supervised_bottleneck_encoder_seed42 --bottleneck-dims 8 16 32 --max-iter 40 --encoder-train-max-segments 80000 --link-max-pairs 2000

.\venv\Scripts\python.exe src\run_supervised_bottleneck_encoder_study.py --max-chunks 0 --seeds 123 456 --run-name supervised_bottleneck_encoder_seeds123456 --bottleneck-dims 8 16 32 --max-iter 40 --encoder-train-max-segments 80000 --link-max-pairs 2000
```

The consolidated artefacts are:

- `outputs/tables/supervised_bottleneck_encoder_allseeds_metrics.csv`
- `outputs/tables/supervised_bottleneck_encoder_allseeds_summary.csv`
- `outputs/tables/privacy_frontier_with_supervised_encoder_summary.csv`
- `outputs/figures/privacy_frontier_with_supervised_encoder.png`

The encoder is fitted only on the train split, using a balanced train-segment sample, and the final hidden bottleneck activation is used as the representation for utility and linkability.

## Encoder plus DP-calibrated noise

DP directly on the 208-feature representation is very noisy, so the current combined experiment applies a supervised bottleneck encoder first and then applies Gaussian noise calibrated by `epsilon`, `delta`, and an L2 clipping norm in the 8-dimensional embedding space:

```powershell
.\venv\Scripts\python.exe src\run_encoder_dp_study.py --max-chunks 0 --seeds 42 --run-name encoder_dp_dim8_seed42 --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --link-max-pairs 2000 --dp-epsilons 100 50 20 --dp-clip-norms 2 4
```

The current artefacts are:

- `outputs/tables/encoder_dp_dim8_seed42_metrics.csv`
- `outputs/tables/encoder_dp_dim8_seed42_summary.csv`
- `outputs/tables/encoder_dp_dim8_seed42_compact_summary.csv`
- `outputs/tables/encoder_dp_dim8_seed42_protocol.json`
- `outputs/figures/encoder_dp_dim8_seed42_tradeoff.png`

Important limitation: this is DP-calibrated perturbation of released embeddings, not DP-SGD. It does not make the encoder training procedure itself differentially private.

Use this command to evaluate the same encoder+DP configurations under the operational-style gallery linkage protocol:

```powershell
.\venv\Scripts\python.exe src\run_encoder_dp_operational_linkage_eval.py --max-chunks 0 --seeds 42 --run-name encoder_dp_operational_dim8_seed42 --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --train-max-pairs 2000 --gallery-negatives 99 999 --queries-per-seed 1000 --dp-configs 100:2 50:2 50:4 20:4
```

Use this command for the final multi-seed encoder+DP operational check:

```powershell
.\venv\Scripts\python.exe src\run_encoder_dp_operational_linkage_eval.py --max-chunks 0 --seeds 42 123 456 --run-name encoder_dp_operational_dim8_multiseed --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --train-max-pairs 2000 --gallery-negatives 99 999 --queries-per-seed 1000 --dp-configs 100:2 50:2 50:4 20:4
```

The current operational encoder+DP artefacts are:

- `outputs/tables/encoder_dp_operational_dim8_seed42_metrics.csv`
- `outputs/tables/encoder_dp_operational_dim8_seed42_summary.csv`
- `outputs/tables/encoder_dp_operational_dim8_seed42_compact_summary.csv`
- `outputs/tables/encoder_dp_operational_dim8_seed42_protocol.json`
- `outputs/tables/encoder_dp_operational_dim8_multiseed_metrics.csv`
- `outputs/tables/encoder_dp_operational_dim8_multiseed_summary.csv`
- `outputs/tables/encoder_dp_operational_dim8_multiseed_protocol.json`
- `outputs/tables/operational_linkage_encoder_dp_comparison.csv`
- `outputs/figures/encoder_dp_operational_dim8_seed42.png`

## Final representation trade-off tables

Use this command to regenerate the final thesis-facing tables and figures:

```powershell
.\venv\Scripts\python.exe src\consolidate_final_representation_results.py
```

The final consolidated artefacts are:

- `outputs/tables/final_representation_tradeoff_multiseed.csv`
- `outputs/tables/final_encoder_dp_multiseed_summary.csv`
- `outputs/tables/final_thesis_ready_representation_table.csv`
- `outputs/tables/final_table_latex.tex`
- `outputs/tables/final_methodological_notes.md`
- `outputs/tables/thesis_methods_draft.md`
- `outputs/tables/results_narrative.md`
- `outputs/tables/thesis_discussion_draft.md`
- `outputs/tables/figure_captions.md`
- `outputs/tables/claims_vs_evidence.md`
- `outputs/figures/final_representation_pairwise_tradeoff_multiseed.png`
- `outputs/figures/final_encoder_dp_operational_multiseed.png`

## Prism thesis handoff

Use these files to give the final project context to Prism or another thesis-writing assistant:

- `outputs/prism_handoff/README.md`
- `outputs/prism_handoff/PRISM_MASTER_PROMPT.md`
- `outputs/prism_handoff/SECTION_PROMPTS.md`

## Representation dimension benchmark

Use these commands to reproduce the seed-42 comparison between supervised encoder, PCA, random projection, and top-k utility features at matched dimensions:

```powershell
.\venv\Scripts\python.exe src\run_representation_dimension_comparison.py --max-chunks 0 --seeds 42 --run-name representation_dimension_comparison_seed42 --dims 2 4 8 16 32 64 --methods identity supervised_encoder pca random_projection topk_utility --encoder-max-iter 40 --encoder-train-max-segments 80000 --link-max-pairs 2000

.\venv\Scripts\python.exe src\run_representation_dimension_comparison.py --max-chunks 0 --seeds 42 --run-name representation_dimension_comparison_seed42_rp_topk --dims 2 4 8 16 32 64 --methods random_projection topk_utility --link-max-pairs 2000
```

The first command writes incrementally and may stop at the environment timeout; the second command was used to complete the random-projection and top-k rows. The consolidated artefacts are:

- `outputs/tables/representation_dimension_comparison_seed42_full_metrics.csv`
- `outputs/tables/representation_dimension_comparison_seed42_full_summary.csv`
- `outputs/figures/representation_dimension_comparison_seed42_full.png`
- `outputs/figures/representation_dimension_frontier_seed42_full.png`

For the thesis-facing robustness rerun, evaluate the retained representation families over three retained-record-level splits:

```powershell
.\venv\Scripts\python.exe src\run_representation_dimension_comparison.py --max-chunks 0 --seeds 42 123 456 --run-name representation_dimension_comparison_multiseed --dims 8 64 --methods identity supervised_encoder pca random_projection --encoder-max-iter 40 --encoder-train-max-segments 80000 --link-max-pairs 2000
```

The runner asserts that retained record identifiers are disjoint across train and test, and exports the exact group splits plus segment-level utility predictions for uncertainty analysis.

The completed consolidated artefacts are:

- `outputs/tables/representation_dimension_final_multiseed_metrics.csv`
- `outputs/tables/representation_dimension_final_multiseed_summary.csv`
- `outputs/tables/representation_dimension_final_multiseed_utility_predictions.csv`
- `outputs/tables/representation_dimension_final_multiseed_splits.csv`
- `outputs/tables/representation_dimension_final_multiseed_protocol.json`

## Aggregate-view linkability

Use this command to reproduce the current seed-42 aggregate-view check:

```powershell
.\venv\Scripts\python.exe src\run_aggregate_linkability_study.py --max-chunks 0 --seeds 42 --run-name aggregate_linkability_full_seed42_mean --protocols odd_even early_late_gap random_halves --stats mean --link-max-pairs 2000
```

The canonical artefacts are:

- `outputs/tables/aggregate_linkability_full_seed42_mean_metrics.csv`
- `outputs/tables/aggregate_linkability_full_seed42_mean_summary.csv`
- `outputs/tables/aggregate_linkability_full_seed42_mean_aggregate_shapes.csv`
- `outputs/tables/aggregate_linkability_full_seed42_mean_protocol.json`
- `outputs/tables/aggregate_linkability_seed42_comparison.csv`
- `outputs/figures/aggregate_linkability_seed42_comparison.png`

This experiment evaluates pairs of aggregate views rather than pairs of individual segments.

## Reproducible transformation study

Use this command for the full final check across repeated retained-record-level splits:

```powershell
.\venv\Scripts\python.exe src\run_privacy_transform_study.py --suite full --seeds 42 123 456 --run-name privacy_transform_full_final --save-fitted
```

The script writes per-seed metrics, aggregate metrics, a protocol JSON, and optional fitted transform objects under `outputs/`.
The canonical final artefacts are:

- `outputs/tables/privacy_transform_full_final_metrics.csv`
- `outputs/tables/privacy_transform_full_final_summary.csv`
- `outputs/tables/privacy_transform_full_final_protocol.json`
- `outputs/tables/privacy_transform_fitted_parameters.csv`
- `outputs/tables/privacy_transform_pca_explained_variance.csv`
- `outputs/tables/privacy_transform_random_projection_parameters.csv`
- `outputs/models/privacy_transform_study/privacy_transform_full_final/`

Use this command for the gallery-style linkage check with synthetic low-prevalence candidate sets:

```powershell
.\venv\Scripts\python.exe src\run_operational_linkage_eval.py --seeds 42 123 456 --gallery-negatives 99 999 --queries-per-seed 1000 --run-name operational_linkage_final
```

The canonical operational linkage artefacts are:

- `outputs/tables/operational_linkage_final_metrics.csv`
- `outputs/tables/operational_linkage_final_summary.csv`
- `outputs/tables/operational_linkage_final_protocol.json`

Use this command for the same gallery-style linkage check on the supervised bottleneck encoder representation:

```powershell
.\venv\Scripts\python.exe src\run_encoder_operational_linkage_eval.py --max-chunks 0 --seeds 42 --run-name encoder_operational_linkage_dim8_seed42 --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --train-max-pairs 2000 --gallery-negatives 99 999 --queries-per-seed 1000
```

The current encoder operational artefacts are:

- `outputs/tables/encoder_operational_linkage_dim8_seed42_metrics.csv`
- `outputs/tables/encoder_operational_linkage_dim8_seed42_summary.csv`
- `outputs/tables/encoder_operational_linkage_dim8_seed42_protocol.json`
- `outputs/tables/operational_linkage_representation_comparison.csv`
- `outputs/figures/operational_linkage_representation_comparison.png`

The hard-negative encoder gallery check was run as a reduced 20-chunk stress test because full hard-negative neighbour construction exceeded the interactive timeout:

```powershell
.\venv\Scripts\python.exe src\run_encoder_operational_linkage_eval.py --max-chunks 20 --seeds 42 --run-name encoder_operational_linkage_dim8_hardneg_20chunks_seed42 --bottleneck-dim 8 --encoder-max-iter 25 --encoder-train-max-segments 40000 --train-max-pairs 2000 --train-negative-strategy hard --hard-negative-pool-size 5 --gallery-negatives 99 999 --queries-per-seed 1000
```

## Differential privacy status

The current project has tested noise-based and projection-based privacy transformations, and the final encoder experiment applies DP-calibrated embedding perturbation with explicit `epsilon`, `delta` and clipping parameters. In thesis wording, distinguish this from DP-SGD or end-to-end differentially private training, because the encoder fitting and preprocessing pipeline are not themselves trained under a formal DP accountant.
