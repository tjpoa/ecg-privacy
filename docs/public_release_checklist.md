# Public release checklist

This checklist separates the technical release work from decisions that must
be confirmed by the authors before publication.

## Prepared in the local repository

- [x] Raw ECG files and derived patient-level datasets are excluded by
      `.gitignore`.
- [x] Generated results, fitted models, and internal manuscript handoff files
      are excluded by `.gitignore`.
- [x] Local `output/` and `tmp/` directories are excluded by `.gitignore`.
- [x] Dataset source, expected directory layout, label codes, and
      non-redistribution policy are documented in `data/README.md`.
- [x] Runtime and notebook dependencies are separated into requirements files.
- [x] A release audit and lightweight regression tests are available.
- [x] The recommended five-notebook workflow is documented.
- [x] The repository does not require an absolute local filesystem path for its
      command-line runners.
- [x] The final article protocol is recorded in both machine-readable and
      human-readable forms.
- [x] Full reproduction instructions use the article's 2-second window and
      1-second step.
- [x] The malformed row in the local canonical article CSV has been corrected.
- [x] A curated public snapshot of aggregate article tables is available under
      `reproducibility/`; it contains no record identifiers or waveforms.
- [x] Third-party IEEEtran files and PhysioNet data are explicitly excluded
      from the future author-code licence.

## Must be confirmed before the first public release

- [x] Choose and add the MIT software licence.
- [ ] Confirm the final author list, affiliations, and ownership of the code.
- [ ] Confirm that the manuscript, figures, and third-party IEEE template files
      should be published in this repository or kept in a separate paper
      repository.
- [ ] Decide whether the final derived tables and protocol JSON files should be
      published in a versioned release or deposited in Zenodo.
- [ ] Create a version tag for the exact code used by the manuscript.
- [ ] Replace the manuscript's provisional code-availability wording with the
      final repository URL or DOI.
- [ ] Run `scripts\verify_public_release.py` and the tests from a clean clone.
- [ ] Run `scripts\verify_public_release.py --require-license` before creating
      the release.

## Recommended release contents

The public repository should contain source code, the five workflow notebooks,
documentation, tests, configuration, and a machine-readable version of the
final protocol. It should not contain raw ECG waveforms, local virtual
environments, private thesis material, or unreviewed exploratory outputs.
