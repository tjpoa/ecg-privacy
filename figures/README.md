# Public figures

This folder contains author-created PNG figures that support the public code
release. PDF and LaTeX submission assets are deliberately kept outside this
repository.

| Figure | Generation source |
| --- | --- |
| `main_utility_linkability_tradeoff.png` | `src/generate_main_tradeoff_figure.py` |
| `threat_model_attack_axes.png` | `src/generate_threat_model_figure.py` |
| `ecg_final_representation_pairwise_tradeoff_multiseed.png` | `src/consolidate_final_representation_results.py` |
| `ecg_final_encoder_dp_operational_multiseed.png` | `src/consolidate_final_representation_results.py` |

The remaining two figures are supporting snapshots from the retained analysis:
`ecg_linkability_sensitivity_gap_hardneg.png` and
`ecg_topk_utility_linkability_curve.png`. They are not required by the final
reproduction workflow.
