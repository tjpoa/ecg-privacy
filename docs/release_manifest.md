# Release manifest

This is the recommended scope for the first public code release.

## Include

- `README.md`, `.gitignore`, dependency files, and the selected `LICENSE`;
- `THIRD_PARTY_NOTICES.md`;
- `configs/experiment_config.yaml`;
- `data/README.md` and the empty metadata placeholder only;
- the five output-free workflow notebooks and `notebooks/README.md`;
- `src/`, `scripts/`, `tests/`, and `docs/`;
- `reproducibility/`, containing only the aggregate tables used by the article.

## Exclude

- `data/raw/`, `data/interim/`, and `data/processed/`;
- `.mat`, `.hea`, patient-level tables, predictions, and fitted models;
- `outputs/`, `output/`, `tmp/`, virtual environments, caches, and logs;
- exploratory or archived notebooks;
- private thesis handoff material;
- credentials, local paths, and machine-specific caches.

## Manuscript and third-party files

The recommended default is to release the code independently from the Overleaf
submission package. If the manuscript source is included, the future software
licence must apply only to author-owned code. `IEEEtran.cls` and `IEEEtran.bst`
remain under the LaTeX Project Public License stated in their headers and are
not relicensed by the authors.

Author-created manuscript text and figures should be given an explicit content
licence or placed in a separate paper archive after all authors confirm
ownership and publication-policy compatibility.

## Actions requiring explicit authorisation

Do not run these steps until the authors approve them:

1. stage and commit the release candidate;
2. push it to GitHub;
3. create the `jbhi-submission-v1` tag;
4. create a GitHub release;
5. archive the release in Zenodo;
6. replace the provisional manuscript URL with the final tag URL or DOI.

Suggested commit message: `Prepare reproducible JBHI submission release`.
