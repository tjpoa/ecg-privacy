# Aggregate article results

The `tables/` directory contains the small aggregate result files needed to
trace the values reported in the JBHI manuscript. These files contain means,
standard deviations, confidence intervals, protocol summaries, and model
settings. They do not contain ECG waveforms, retained-record identifiers,
per-record predictions, or fitted models.

## Mapping to the manuscript

- `main_utility_linkability_tradeoff_figure_data.csv`: Table I and Figure 1;
- `main_paired_uncertainty_summary.csv`: Table II;
- `article_canonical_representation_table.csv`: operational gallery values in
  Table III and related representation summaries;
- `absolute_attribute_risk_summary.csv`: Table IV;
- `privacy_attack_extensions_final_protocol_attribute_bootstrap_effects.csv`:
  incremental attribute effects and confidence intervals;
- `privacy_attack_extensions_final_protocol_reconstruction_article_table.csv`:
  Table V;
- `conditioned_reconstruction_final_protocol_multiseed_article_table.csv`:
  linkage-conditioned reconstruction statements;
- `article_configuration_selection_evidence.csv` and
  `article_model_configuration_audit.csv`: configuration-selection evidence
  and fixed model settings;
- `computational_environment_versions.csv`: recorded software environment.

The machine-readable protocol is in `../configs/experiment_config.yaml`, and a
human-readable description is in `../docs/final_protocol.md`.

These aggregate tables are snapshots for `jbhi-submission-v1`. Regenerating
them requires the source PhysioNet data and writes working artefacts to the
ignored local `outputs/` directory.

After regenerating the local tables, refresh this curated snapshot with:

```powershell
python scripts\export_public_aggregate_tables.py
```

The export rejects inconsistent CSV rows and tables containing common
record-level identifier columns.
