# Final article protocol

This document is the human-readable companion to
`configs/experiment_config.yaml`. It describes the protocol used for the JBHI
submission. It does not describe every exploratory experiment in the project.

## Data and preprocessing

The source is the PhysioNet ECG Arrhythmia Database v1.0.0. The repository does
not redistribute its waveform or header files. Of 45,152 source records, 45,151
were retained after processing. Each 10-second, 12-lead recording was sampled
at 500 Hz, filtered from 0.5 to 40 Hz, and standardised by lead within the full
retained record.

Each record was divided into 2-second windows with a 1-second step. This
produced nine segments per record and 406,359 segments in total. The positive
utility class contains SNOMED-CT codes `164889003` and `164890007`. The retained
record counts are 9,840 atrial and 35,311 non-atrial.

## Shared evaluation design

All final analyses use retained-record-level train/test splits. No segment from
a test record is available during model or transformation fitting. The test
fraction is 0.2. The repeated evaluations use seeds 42, 123, and 456.

The main comparison uses the same label-balanced sample of 40,000 training
segments for every representation within a seed. The representations are the
208 handcrafted features, 8-dimensional PCA, 8-dimensional random projection,
an 8-dimensional supervised encoder, and two Gaussian-perturbed variants of
that encoder.

The Gaussian variants use clipping norms 2 and 4. Their noise standard
deviations are 0.3875844210 and 0.7751688420. Both have noise multiplier
0.1937922105. These values specify the empirical release transformation. They
do not establish DP-SGD or end-to-end differential privacy.

## Utility and linkability

Utility is atrial versus non-atrial classification with a logistic-regression
probe. The primary metric is balanced accuracy. Record-level performance is
obtained by averaging the nine segment probabilities for each record.

Pairwise linkability asks whether two segments came from the same retained ECG
record. The attacker is XGBoost on the element-wise absolute representation
difference. Each seed uses 2,000 positive and 2,000 negative training pairs.
Positive pairs have a minimum gap of four segment positions, with at most three
positive pairs per record.

The operational gallery uses 1,000 queries per seed. Each query has one true
same-record candidate and 999 different-record candidates. Its positive
prevalence is therefore 0.001. The reported metrics are PR-AUC, Recall@1, and
mean reciprocal rank.

## Attribute inference and reconstruction

The attribute experiment uses 1,000 anchors per seed. Segment 0 is the anchor.
Segments 4, 6, and 8 are the true same-record candidates, mixed with 99
different-record candidates. Sex, age at least 65, and continuous age are
evaluated. Incremental effects use 2,000 paired bootstrap replicates.

The reconstruction cohort contains 12,000 records. Segment 0 is the target.
The 12-lead target is downsampled to 100 Hz and represented by 128 training-fit
PCA components. Ridge and multilayer-perceptron decoders are evaluated. The
main metrics are percent root-mean-square difference and waveform correlation,
including Lead II correlation after a maximum lag search of 250 ms.

## Interpretation boundary

The identifier used by the implementation denotes a retained ECG record. The
experiments do not establish longitudinal patient identity, cross-session
re-identification resistance, complete anonymisation, or end-to-end
differential privacy.
