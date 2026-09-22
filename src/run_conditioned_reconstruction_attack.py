from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from modeling import get_feature_columns, split_segments_by_patient_stratified
    from run_attribute_inference_attack import (
        build_linkage_attack_views,
        build_true_view,
        train_linkability_attacker,
    )
    from run_encoder_dp_study import apply_dp_to_embedding_split
    from run_operational_linkage_eval import load_feature_dataset
    from run_reconstruction_attack import (
        align_targets,
        build_or_load_target_cohort,
        fit_decoder,
        reconstruction_metrics,
    )
    from run_representation_dimension_comparison import build_supervised_encoder_representation
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from .modeling import get_feature_columns, split_segments_by_patient_stratified
    from .run_attribute_inference_attack import (
        build_linkage_attack_views,
        build_true_view,
        train_linkability_attacker,
    )
    from .run_encoder_dp_study import apply_dp_to_embedding_split
    from .run_operational_linkage_eval import load_feature_dataset
    from .run_reconstruction_attack import (
        align_targets,
        build_or_load_target_cohort,
        fit_decoder,
        reconstruction_metrics,
    )
    from .run_representation_dimension_comparison import build_supervised_encoder_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate waveform reconstruction from an anchor representation alone and after "
            "oracle, attacker-selected, or random segment grouping."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--records-dir", type=Path, default=WFDB_RECORDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--target-cache", type=Path, default=None)
    parser.add_argument("--run-name", default="conditioned_reconstruction_attack")
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--target-records", type=int, default=12000)
    parser.add_argument("--cohort-seed", type=int, default=2026)
    parser.add_argument("--segment-ref", type=int, default=0)
    parser.add_argument("--target-fs", type=int, default=100)
    parser.add_argument("--target-pca-components", type=int, default=128)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--max-lag-ms", type=float, default=250.0)

    parser.add_argument("--group-refs", type=int, nargs="+", default=[0, 4, 6, 8])
    parser.add_argument("--attack-queries", type=int, default=1000)
    parser.add_argument("--gallery-negatives", type=int, default=99)
    parser.add_argument("--train-max-pairs", type=int, default=2000)
    parser.add_argument("--min-segment-gap", type=int, default=4)
    parser.add_argument("--max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)

    parser.add_argument("--decoders", nargs="+", choices=["ridge", "mlp"], default=["ridge", "mlp"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--decoder-hidden-units", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--decoder-max-iter", type=int, default=100)
    parser.add_argument("--decoder-batch-size", type=int, default=256)
    parser.add_argument("--decoder-learning-rate-init", type=float, default=0.001)

    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=80000)
    parser.add_argument("--dp-epsilon", type=float, default=50.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=2.0)
    return parser.parse_args()


def default_target_cache(args: argparse.Namespace) -> Path:
    return DATA_INTERIM / (
        f"reconstruction_targets_ref{args.segment_ref}_{args.target_fs}hz_n{args.target_records}.npz"
    )


def choose_query_ids(
    test_df: pd.DataFrame,
    target_ids: set[str],
    group_refs: list[int],
    n_queries: int,
    seed: int,
) -> list[str]:
    required_refs = {int(value) for value in group_refs}
    selected = test_df[test_df["segment_ref"].astype(int).isin(required_refs)]
    complete = selected.groupby("patient_id")["segment_ref"].nunique()
    candidates = sorted(
        str(patient_id)
        for patient_id, count in complete.items()
        if int(count) == len(required_refs) and str(patient_id) in target_ids
    )
    if not candidates:
        raise ValueError("No reconstruction targets have all required grouping references in the test split.")
    rng = np.random.default_rng(seed)
    selected_count = min(int(n_queries), len(candidates))
    return sorted(rng.choice(np.asarray(candidates, dtype=object), size=selected_count, replace=False).astype(str))


def select_view_arrays(
    view_df: pd.DataFrame,
    feature_columns: list[str],
    allowed_ids: set[str],
    target_ids: np.ndarray,
    target_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = view_df[view_df["patient_id"].astype(str).isin(allowed_ids)].copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    selected = selected.drop_duplicates("patient_id", keep="first").sort_values("patient_id")
    patient_ids = selected["patient_id"].to_numpy(dtype=str)
    if len(patient_ids) == 0:
        raise ValueError("A reconstruction view contains no records with waveform targets.")
    matrix = selected[feature_columns].to_numpy(dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    targets = align_targets(patient_ids, target_ids, target_matrix)
    return patient_ids, matrix, targets


def evaluate_decoder_conditions(
    representation: str,
    train_views: dict[str, pd.DataFrame],
    test_views: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_ids: np.ndarray,
    target_matrix: np.ndarray,
    target_pca: PCA,
    target_scaler: StandardScaler,
    link_metrics: dict,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[pd.DataFrame]]:
    target_set = set(str(value) for value in target_ids)
    test_map = {
        "single_segment": ["single_segment"],
        "oracle_group": ["oracle_group", "link_selected_group", "random_group"],
    }
    metrics_rows: list[dict] = []
    per_record_frames: list[pd.DataFrame] = []

    for train_condition, test_conditions in test_map.items():
        train_ids, X_train, y_train = select_view_arrays(
            train_views[train_condition],
            feature_columns=feature_columns,
            allowed_ids=target_set,
            target_ids=target_ids,
            target_matrix=target_matrix,
        )
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
        target_components_train = target_scaler.transform(target_pca.transform(y_train))

        for decoder_name in args.decoders:
            model = fit_decoder(
                decoder_name=decoder_name,
                X_train=X_train_scaled,
                target_components_train=target_components_train,
                seed=seed,
                args=args,
            )
            model_n_iter = getattr(model, "n_iter_", None)
            decoder_n_iter = (
                int(np.asarray(model_n_iter).reshape(-1)[0]) if model_n_iter is not None else np.nan
            )

            for test_condition in test_conditions:
                test_ids, X_test, y_test = select_view_arrays(
                    test_views[test_condition],
                    feature_columns=feature_columns,
                    allowed_ids=target_set,
                    target_ids=target_ids,
                    target_matrix=target_matrix,
                )
                X_test_scaled = scaler.transform(imputer.transform(X_test))
                predicted_components = target_scaler.inverse_transform(model.predict(X_test_scaled))
                y_pred = target_pca.inverse_transform(predicted_components).astype(np.float32, copy=False)
                metrics, per_record = reconstruction_metrics(
                    patient_ids=test_ids,
                    y_true_flat=y_test,
                    y_pred_flat=y_pred,
                    n_leads=12,
                    target_fs=args.target_fs,
                    max_lag_ms=args.max_lag_ms,
                )
                metrics_rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "decoder": decoder_name,
                        "train_condition": train_condition,
                        "test_condition": test_condition,
                        "n_train": int(len(train_ids)),
                        "n_features": int(len(feature_columns)),
                        "decoder_n_iter": decoder_n_iter,
                        "link_selection_precision": float(link_metrics["selection_precision"]),
                        "link_exact_group_recovery": float(link_metrics["exact_group_recovery"]),
                        **metrics,
                    }
                )
                per_record.insert(0, "test_condition", test_condition)
                per_record.insert(0, "train_condition", train_condition)
                per_record.insert(0, "decoder", decoder_name)
                per_record.insert(0, "representation", representation)
                per_record.insert(0, "seed", seed)
                per_record_frames.append(per_record)
    return metrics_rows, per_record_frames


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["representation", "decoder", "train_condition", "test_condition"]
    metric_columns = [
        "n_train",
        "n_test",
        "link_selection_precision",
        "link_exact_group_recovery",
        "rmse_mean",
        "prd_mean",
        "snr_db_mean",
        "mean_lead_correlation_mean",
        "lead_ii_correlation_mean",
        "lead_ii_maxlag_correlation_mean",
        "lead_ii_spectral_correlation_mean",
    ]
    summary = metrics_df.groupby(group_columns, as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in column if part]) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    if len(args.group_refs) < 2 or args.group_refs[0] != args.segment_ref:
        raise ValueError("--group-refs must start with --segment-ref and include at least one candidate reference.")
    if len(set(args.group_refs)) != len(args.group_refs):
        raise ValueError("--group-refs must contain unique segment references.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    features_df["patient_id"] = features_df["patient_id"].astype(str)

    target_cache = args.target_cache or default_target_cache(args)
    print("Preparing waveform target cohort...")
    target_ids, target_matrix, target_metadata = build_or_load_target_cohort(
        features_df=features_df,
        manifest=manifest,
        records_dir=args.records_dir,
        cache_path=target_cache,
        target_records=args.target_records,
        cohort_seed=args.cohort_seed,
        segment_ref=args.segment_ref,
        target_fs=args.target_fs,
        workers=args.target_workers,
    )
    target_set = set(str(value) for value in target_ids)

    metrics_rows: list[dict] = []
    per_record_frames: list[pd.DataFrame] = []
    link_rows: list[dict] = []
    pca_rows: list[dict] = []
    encoder_protocols: list[dict] = []
    noise_protocols: list[dict] = []

    for seed in args.seeds:
        print(f"Seed {seed}: creating record-disjoint split...")
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        query_ids = choose_query_ids(
            test_df=test_df,
            target_ids=target_set,
            group_refs=args.group_refs,
            n_queries=args.attack_queries,
            seed=seed,
        )

        train_target_ids = np.asarray(
            sorted(set(train_df["patient_id"].astype(str)).intersection(target_set)), dtype=str
        )
        y_train_all = align_targets(train_target_ids, target_ids, target_matrix)
        n_components = min(args.target_pca_components, len(y_train_all) - 1, y_train_all.shape[1])
        target_pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=3,
            random_state=seed,
        )
        target_components_train = target_pca.fit_transform(y_train_all)
        target_scaler = StandardScaler().fit(target_components_train)
        pca_rows.append(
            {
                "seed": seed,
                "n_components": int(n_components),
                "explained_variance_ratio_sum": float(target_pca.explained_variance_ratio_.sum()),
                "train_targets": int(len(y_train_all)),
                "conditioned_test_queries": int(len(query_ids)),
            }
        )

        representation_splits = {"raw_features": (train_df, test_df)}
        print(f"Seed {seed}: fitting {args.bottleneck_dim}-D utility encoder...")
        feature_columns = get_feature_columns(features_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_encoder_df, test_encoder_df, encoder_metadata = build_supervised_encoder_representation(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                dim=args.bottleneck_dim,
                seed=seed,
                args=args,
            )
        encoder_protocols.append({"seed": seed, **encoder_metadata})
        representation_splits["encoder"] = (train_encoder_df, test_encoder_df)

        train_noisy_df, test_noisy_df, noise_metadata = apply_dp_to_embedding_split(
            train_embedding_df=train_encoder_df,
            test_embedding_df=test_encoder_df,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            clip_norm=args.dp_clip_norm,
            seed=seed + 30_000,
        )
        noise_protocols.append({"seed": seed, **noise_metadata})
        representation_splits["encoder_gaussian_noise"] = (train_noisy_df, test_noisy_df)

        for representation, (train_representation_df, test_representation_df) in representation_splits.items():
            print(f"Seed {seed}: building conditioned reconstruction views for {representation}...")
            feature_columns = get_feature_columns(train_representation_df)
            link_model, pair_scaler, train_pair_df = train_linkability_attacker(
                train_representation_df,
                feature_columns=feature_columns,
                seed=seed,
                args=args,
            )
            test_views, link_metrics = build_linkage_attack_views(
                test_representation_df,
                feature_columns=feature_columns,
                query_ids=query_ids,
                group_refs=args.group_refs,
                gallery_negatives=args.gallery_negatives,
                model=link_model,
                scaler=pair_scaler,
                seed=seed + 20_000,
            )
            train_views = {
                "single_segment": build_true_view(
                    train_representation_df, feature_columns, refs=[args.segment_ref]
                ),
                "oracle_group": build_true_view(
                    train_representation_df, feature_columns, refs=args.group_refs
                ),
            }
            rows, records = evaluate_decoder_conditions(
                representation=representation,
                train_views=train_views,
                test_views=test_views,
                feature_columns=feature_columns,
                target_ids=target_ids,
                target_matrix=target_matrix,
                target_pca=target_pca,
                target_scaler=target_scaler,
                link_metrics=link_metrics,
                seed=seed,
                args=args,
            )
            metrics_rows.extend(rows)
            per_record_frames.extend(records)
            link_rows.append(
                {
                    "seed": seed,
                    "representation": representation,
                    "train_pairs": int(len(train_pair_df)),
                    "train_pair_positive_prevalence": float(train_pair_df["pair_label"].mean()),
                    **link_metrics,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    per_record_df = pd.concat(per_record_frames, ignore_index=True)
    link_df = pd.DataFrame(link_rows)
    pca_df = pd.DataFrame(pca_rows)
    summary_df = summarize_metrics(metrics_df)
    article_table_df = summary_df.loc[
        summary_df["decoder"].eq("mlp"),
        [
            "representation",
            "test_condition",
            "prd_mean_mean",
            "prd_mean_std",
            "mean_lead_correlation_mean_mean",
            "mean_lead_correlation_mean_std",
        ],
    ].sort_values(["representation", "test_condition"])

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    per_record_path = args.output_dir / f"{args.run_name}_per_record.csv.gz"
    link_path = args.output_dir / f"{args.run_name}_link_selection.csv"
    pca_path = args.output_dir / f"{args.run_name}_target_pca.csv"
    article_table_path = args.output_dir / f"{args.run_name}_article_table.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    per_record_df.to_csv(per_record_path, index=False, compression="gzip")
    link_df.to_csv(link_path, index=False)
    pca_df.to_csv(pca_path, index=False)
    article_table_df.to_csv(article_table_path, index=False)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "records_dir": str(args.records_dir),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": int(len(features_df)),
        "loaded_identifiers": int(features_df["patient_id"].nunique()),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "split_unit": "retained record identifier",
        "target_cohort": target_metadata,
        "waveform_target": {
            "segment_ref": args.segment_ref,
            "target_sampling_rate_hz": args.target_fs,
            "leads": 12,
            "target_pca_components": args.target_pca_components,
        },
        "conditioned_attack": {
            "group_refs": args.group_refs,
            "aggregation": "feature-wise mean",
            "queries_per_seed": args.attack_queries,
            "gallery_negatives": args.gallery_negatives,
            "true_candidates_per_query": len(args.group_refs) - 1,
            "single_decoder_training": "segment-ref anchor representations",
            "group_decoder_training": "oracle same-record aggregates",
            "group_decoder_tests": ["oracle_group", "link_selected_group", "random_group"],
            "reconstruction_target": "anchor segment waveform",
        },
        "linkability_attacker": {
            "model": "XGBoost on absolute representation differences",
            "train_max_pairs": args.train_max_pairs,
            "min_segment_gap": args.min_segment_gap,
            "max_positive_pairs_per_identifier": args.max_positive_pairs_per_patient,
        },
        "decoders": {
            "models": args.decoders,
            "ridge_alpha": args.ridge_alpha,
            "mlp_hidden_units": args.decoder_hidden_units,
            "mlp_max_iter": args.decoder_max_iter,
        },
        "encoder": {
            "bottleneck_dim": args.bottleneck_dim,
            "hidden_units": args.encoder_hidden_units,
            "max_iter": args.encoder_max_iter,
            "train_max_segments": args.encoder_train_max_segments,
            "per_seed": encoder_protocols,
        },
        "gaussian_noise": {
            "epsilon_argument": args.dp_epsilon,
            "delta_argument": args.dp_delta,
            "clip_norm": args.dp_clip_norm,
            "scope_note": (
                "The experiment reuses clipped standardized embeddings plus Gaussian noise. "
                "It is not evidence of end-to-end differential privacy."
            ),
            "per_seed": noise_protocols,
        },
        "claim_boundary": (
            "The grouped attack tests whether attacker-selected same-record aggregation changes "
            "anchor-waveform reconstruction under the defined same-retained-record protocol. "
            "It does not establish cross-session identity recovery or universal reconstruction resistance."
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "per_record": str(per_record_path),
            "link_selection": str(link_path),
            "target_pca": str(pca_path),
            "article_table": str(article_table_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved protocol: {protocol_path}")


if __name__ == "__main__":
    main()
