# JBHI ECG manuscript

`overleaf_jbhi_submission/main.tex` is the canonical self-contained Overleaf
source for the Study II ECG article. `jbhi_ecg_article.tex` mirrors that source
for use from the repository-level manuscript directory.

Regenerate and verify the mirror with:

```powershell
.\venv\Scripts\python.exe .\scripts\sync_manuscript_source.py
.\venv\Scripts\python.exe .\scripts\sync_manuscript_source.py --check
```

Before submission:

1. Confirm the author order and corresponding-author details.
2. Verify that every table matches the final 40,000-segment/25-iteration
   protocol outputs.
3. Compile in Overleaf and confirm the eight-page limit and float placement.
4. Complete funding, conflicts, acknowledgments, and the software licence.
5. Tag the public code version and update the availability statement if a
   Zenodo DOI is created.

Compile with the IEEEtran class in the submission package and run
LaTeX/BibTeX enough times to resolve citations and references. The final local
audit compiled the submission in eight Letter pages without undefined
citations or references.
