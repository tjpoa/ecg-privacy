from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:  # pragma: no cover - required by the project
    raise ImportError("xgboost is required for the attribute-inference attack.") from exc

try:
    from config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from modeling import _build_xgb_model, build_pair_table, get_feature_columns, split_segments_by_patient_stratified
    from run_encoder_dp_study import apply_dp_to_embedding_split
    from run_operational_linkage_eval import build_absdiff_pair_features_fast, load_feature_dataset
    from run_representation_dimension_comparison import build_supervised_encoder_representation
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from .modeling import (
        _build_xgb_model,
        build_pair_table,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )
    from .run_encoder_dp_study import apply_dp_to_embedding_split
    from .run_operational_linkage_eval import build_absdiff_pair_features_fast, load_feature_dataset
    from .run_representation_dimension_comparison import build_supervised_encoder_representation


DEFAULT_METADATA_CACHE = DATA_INTERIM / "demographic_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether same-record segment linkage increases inference of sex and age "
            "from released ECG representations."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--records-dir", type=Path, default=WFDB_RECORDS_DIR)
    parser.add_argument("--metadata-cache", type=Path, default=DEFAULT_METADATA_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--figure-dir", type=Path, default=OUTPUTS_FIGURES_DIR)
    parser.add_argument("--run-name", default="attribute_inference_attack")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of feature chunks to load. Use 0 for the complete dataset.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--group-refs", type=int, nargs="+", default=[0, 4, 6, 8])
    parser.add_argument("--attack-queries", type=int, default=1000)
    parser.add_argument("--gallery-negatives", type=int, default=99)
    parser.add_argument("--train-max-pairs", type=int, default=2000)
    parser.add_argument("--min-segment-gap", type=int, default=4)
    parser.add_argument("--max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=80000)
    parser.add_argument("--dp-epsilon", type=float, default=50.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=2.0)
    parser.add_argument("--attribute-estimators", type=int, default=200)
    parser.add_argument("--metadata-workers", type=int, default=16)
    return parser.parse_args()


def normalize_sex(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"male", "m"}:
        return "Male"
    if normalized in {"female", "f"}:
        return "Female"
    return None


def parse_demographic_header(path: Path) -> dict:
    age = np.nan
    sex = None
    diagnosis = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#Age:"):
                value = line.split(":", 1)[1].strip()
                try:
                    parsed_age = float(value)
                    age = parsed_age if 0.0 <= parsed_age <= 120.0 else np.nan
                except ValueError:
                    age = np.nan
            elif line.startswith("#Sex:"):
                sex = normalize_sex(line.split(":", 1)[1].strip())
            elif line.startswith("#Dx:"):
                diagnosis = line.split(":", 1)[1].strip()
    return {
        "patient_id": path.stem,
        "age": age,
        "sex": sex,
        "diagnosis": diagnosis,
    }


def load_demographic_metadata(
    records_dir: Path,
    cache_path: Path,
    required_ids: set[str],
    workers: int,
) -> tuple[pd.DataFrame, dict]:
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, low_memory=False)
        cache_df["patient_id"] = cache_df["patient_id"].astype(str)
    else:
        cache_df = pd.DataFrame(columns=["patient_id", "age", "sex", "diagnosis"])

    cached_ids = set(cache_df["patient_id"].astype(str))
    missing_ids = required_ids - cached_ids
    parsed_rows: list[dict] = []

    if missing_ids:
        header_paths = [path for path in records_dir.rglob("*.hea") if path.stem in missing_ids]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                parsed_rows = list(executor.map(parse_demographic_header, header_paths))
        else:
            parsed_rows = [parse_demographic_header(path) for path in header_paths]

        if parsed_rows:
            parsed_df = pd.DataFrame(parsed_rows)
            cache_df = parsed_df if cache_df.empty else pd.concat([cache_df, parsed_df], ignore_index=True)
            cache_df = cache_df.drop_duplicates("patient_id", keep="last").sort_values("patient_id")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(cache_path, index=False)

    selected = cache_df[cache_df["patient_id"].isin(required_ids)].copy()
    selected["age"] = pd.to_numeric(selected["age"], errors="coerce")
    selected["sex"] = selected["sex"].map(normalize_sex)
    coverage = {
        "required_identifiers": int(len(required_ids)),
        "metadata_rows_found": int(selected["patient_id"].nunique()),
        "age_available": int(selected["age"].notna().sum()),
        "sex_available": int(selected["sex"].notna().sum()),
        "new_headers_parsed": int(len(parsed_rows)),
        "cache_path": str(cache_path),
    }
    return selected, coverage


def build_true_view(
    representation_df: pd.DataFrame,
    feature_columns: list[str],
    refs: list[int],
) -> pd.DataFrame:
    refs = [int(ref) for ref in refs]
    selected = representation_df[representation_df["segment_ref"].astype(int).isin(refs)].copy()
    counts = selected.groupby("patient_id")["segment_ref"].nunique()
    valid_ids = counts[counts == len(set(refs))].index
    selected = selected[selected["patient_id"].isin(valid_ids)]

    metadata = (
        selected.groupby("patient_id", as_index=False)
        .agg(utility_label=("utility_label", "first"))
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    feature_view = selected.groupby("patient_id", as_index=False)[feature_columns].mean()
    return metadata.merge(feature_view, on="patient_id", how="inner")


def build_record_ref_index(representation_df: pd.DataFrame, required_refs: list[int]) -> dict[str, dict[int, int]]:
    selected = representation_df[
        representation_df["segment_ref"].astype(int).isin([int(ref) for ref in required_refs])
    ]
    mapping: dict[str, dict[int, int]] = {}
    for row_index, patient_id, segment_ref in selected[["patient_id", "segment_ref"]].itertuples():
        mapping.setdefault(str(patient_id), {})[int(segment_ref)] = int(row_index)
    return mapping


def choose_query_ids(
    test_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    group_refs: list[int],
    n_queries: int,
    seed: int,
) -> list[str]:
    counts = (
        test_df[test_df["segment_ref"].astype(int).isin(group_refs)]
        .groupby("patient_id")["segment_ref"]
        .nunique()
    )
    valid_ids = set(str(value) for value in counts[counts == len(set(group_refs))].index)
    demographic_ids = set(
        demographics_df.loc[
            demographics_df["age"].notna() | demographics_df["sex"].notna(), "patient_id"
        ].astype(str)
    )
    candidates = np.asarray(sorted(valid_ids.intersection(demographic_ids)), dtype=object)
    if len(candidates) == 0:
        raise ValueError("No test identifiers have all requested segment references and demographic labels.")
    rng = np.random.default_rng(seed)
    if n_queries > 0 and n_queries < len(candidates):
        candidates = rng.choice(candidates, size=n_queries, replace=False)
    return [str(value) for value in candidates]


def clean_pair_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def train_linkability_attacker(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    args: argparse.Namespace,
):
    pair_df = build_pair_table(
        train_df,
        max_pairs=args.train_max_pairs,
        random_state=seed,
        min_segment_gap=args.min_segment_gap,
        max_pairs_per_patient=args.max_positive_pairs_per_patient,
        negative_strategy="random",
        hard_negative_pool_size=args.hard_negative_pool_size,
    )
    X_pair, y_pair = build_absdiff_pair_features_fast(train_df, pair_df, feature_columns)
    X_pair = clean_pair_matrix(X_pair)
    scaler = StandardScaler()
    X_pair = scaler.fit_transform(X_pair)
    model = _build_xgb_model(random_state=seed, n_jobs=4)
    model.fit(X_pair, y_pair)
    return model, scaler, pair_df


def score_link_candidates(
    model,
    scaler: StandardScaler,
    representation_matrix: np.ndarray,
    anchor_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    pair_matrix = np.abs(representation_matrix[anchor_indices] - representation_matrix[candidate_indices])
    pair_matrix = clean_pair_matrix(pair_matrix)
    pair_matrix = scaler.transform(pair_matrix)
    return model.predict_proba(pair_matrix)[:, list(model.classes_).index(1)]


def build_linkage_attack_views(
    test_df: pd.DataFrame,
    feature_columns: list[str],
    query_ids: list[str],
    group_refs: list[int],
    gallery_negatives: int,
    model,
    scaler: StandardScaler,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if len(group_refs) < 2:
        raise ValueError("group_refs must contain an anchor and at least one additional segment.")

    group_refs = [int(ref) for ref in group_refs]
    anchor_ref = group_refs[0]
    positive_refs = group_refs[1:]
    record_ref_index = build_record_ref_index(test_df, group_refs)
    complete_ids = sorted(
        patient_id
        for patient_id, refs in record_ref_index.items()
        if all(ref in refs for ref in group_refs)
    )
    complete_set = set(complete_ids)
    query_ids = [patient_id for patient_id in query_ids if patient_id in complete_set]
    if not query_ids:
        raise ValueError("No selected queries have complete requested segment references.")

    representation_matrix = test_df[feature_columns].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    n_to_select = len(positive_refs)

    view_rows = {"single_segment": [], "oracle_group": [], "link_selected_group": [], "random_group": []}
    true_selected_counts = []
    reciprocal_ranks = []
    all_true_recovered = []
    candidate_count = n_to_select + gallery_negatives

    for patient_id in query_ids:
        own_refs = record_ref_index[patient_id]
        anchor_index = own_refs[anchor_ref]
        true_indices = np.asarray([own_refs[ref] for ref in positive_refs], dtype=int)

        negative_pool = np.asarray([value for value in complete_ids if value != patient_id], dtype=object)
        replace = gallery_negatives > len(negative_pool)
        negative_ids = rng.choice(negative_pool, size=gallery_negatives, replace=replace)
        negative_indices = np.asarray(
            [record_ref_index[str(negative_id)][int(rng.choice(positive_refs))] for negative_id in negative_ids],
            dtype=int,
        )

        candidate_indices = np.concatenate([true_indices, negative_indices])
        candidate_is_true = np.concatenate(
            [np.ones(len(true_indices), dtype=bool), np.zeros(len(negative_indices), dtype=bool)]
        )
        permutation = rng.permutation(len(candidate_indices))
        candidate_indices = candidate_indices[permutation]
        candidate_is_true = candidate_is_true[permutation]

        anchor_indices = np.full(len(candidate_indices), anchor_index, dtype=int)
        scores = score_link_candidates(
            model=model,
            scaler=scaler,
            representation_matrix=representation_matrix,
            anchor_indices=anchor_indices,
            candidate_indices=candidate_indices,
        )
        ranking = np.argsort(scores)[::-1]
        selected_positions = ranking[:n_to_select]
        selected_indices = candidate_indices[selected_positions]
        selected_true = candidate_is_true[selected_positions]

        true_ranks = np.flatnonzero(candidate_is_true[ranking]) + 1
        reciprocal_ranks.append(float(np.mean(1.0 / true_ranks)))
        true_selected_counts.append(int(selected_true.sum()))
        all_true_recovered.append(float(selected_true.all()))

        random_negative_positions = np.flatnonzero(~candidate_is_true)
        random_positions = rng.choice(random_negative_positions, size=n_to_select, replace=False)
        random_indices = candidate_indices[random_positions]

        single_vector = representation_matrix[anchor_index]
        oracle_vector = np.nanmean(representation_matrix[np.concatenate([[anchor_index], true_indices])], axis=0)
        linked_vector = np.nanmean(representation_matrix[np.concatenate([[anchor_index], selected_indices])], axis=0)
        random_vector = np.nanmean(representation_matrix[np.concatenate([[anchor_index], random_indices])], axis=0)
        utility_label = str(test_df.iloc[anchor_index]["utility_label"])

        for condition, vector in {
            "single_segment": single_vector,
            "oracle_group": oracle_vector,
            "link_selected_group": linked_vector,
            "random_group": random_vector,
        }.items():
            row = {"patient_id": patient_id, "utility_label": utility_label}
            row.update({feature: float(value) for feature, value in zip(feature_columns, vector)})
            view_rows[condition].append(row)

    views = {condition: pd.DataFrame(rows) for condition, rows in view_rows.items()}
    link_metrics = {
        "queries": int(len(query_ids)),
        "group_size": int(len(group_refs)),
        "true_candidates_per_query": int(n_to_select),
        "gallery_negatives": int(gallery_negatives),
        "candidates_per_query": int(candidate_count),
        "selected_true_mean": float(np.mean(true_selected_counts)),
        "selection_precision": float(np.mean(true_selected_counts) / n_to_select),
        "selection_recall": float(np.mean(true_selected_counts) / n_to_select),
        "exact_group_recovery": float(np.mean(all_true_recovered)),
        "multi_positive_mrr": float(np.mean(reciprocal_ranks)),
    }
    return views, link_metrics


def build_attribute_model(task: str, seed: int, n_estimators: int, y_train: np.ndarray):
    common = {
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    if task in {"sex", "age_group"}:
        positives = max(int(np.sum(y_train == 1)), 1)
        negatives = max(int(np.sum(y_train == 0)), 1)
        return XGBClassifier(
            **common,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / positives,
        )
    if task == "age":
        return XGBRegressor(**common, objective="reg:squarederror", eval_metric="rmse")
    raise ValueError(f"Unsupported task: {task}")


def prepare_attribute_data(
    view_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    feature_columns: list[str],
    task: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged = view_df.merge(demographics_df[["patient_id", "age", "sex"]], on="patient_id", how="inner")
    if task == "sex":
        merged = merged[merged["sex"].isin(["Male", "Female"])].copy()
        target = (merged["sex"] == "Female").to_numpy(dtype=int)
    elif task == "age_group":
        merged = merged[merged["age"].notna()].copy()
        target = (merged["age"] >= 65.0).to_numpy(dtype=int)
    elif task == "age":
        merged = merged[merged["age"].notna()].copy()
        target = merged["age"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported task: {task}")
    matrix = merged[feature_columns].to_numpy(dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return merged, matrix, target


def evaluate_task_predictions(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_score=None) -> dict:
    if task in {"sex", "age_group"}:
        return {
            "n_test": int(len(y_true)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
            "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        }
    correlation = (
        spearmanr(y_true, y_pred).statistic
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else np.nan
    )
    return {
        "n_test": int(len(y_true)),
        "mae_years": float(mean_absolute_error(y_true, y_pred)),
        "rmse_years": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_r": float(correlation),
    }


def fit_and_evaluate_attribute_model(
    task: str,
    train_view: pd.DataFrame,
    test_views: dict[str, pd.DataFrame],
    demographics_df: pd.DataFrame,
    feature_columns: list[str],
    representation: str,
    train_condition: str,
    seed: int,
    n_estimators: int,
) -> tuple[list[dict], list[dict]]:
    train_merged, X_train, y_train = prepare_attribute_data(
        train_view, demographics_df, feature_columns, task
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    model = build_attribute_model(task, seed=seed, n_estimators=n_estimators, y_train=y_train)
    model.fit(X_train, y_train)

    metrics_rows = []
    prediction_rows = []
    for condition, test_view in test_views.items():
        test_merged, X_test, y_test = prepare_attribute_data(
            test_view, demographics_df, feature_columns, task
        )
        X_test = imputer.transform(X_test)
        y_pred = model.predict(X_test)
        if task in {"sex", "age_group"}:
            y_pred = np.asarray(y_pred, dtype=int)
            y_score = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        else:
            y_score = None

        metrics = evaluate_task_predictions(task, y_test, y_pred, y_score=y_score)
        metrics_rows.append(
            {
                "seed": seed,
                "task": task,
                "representation": representation,
                "train_condition": train_condition,
                "test_condition": condition,
                "n_train": int(len(train_merged)),
                **metrics,
            }
        )
        for row_index, patient_id in enumerate(test_merged["patient_id"].astype(str)):
            prediction_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "representation": representation,
                    "train_condition": train_condition,
                    "test_condition": condition,
                    "patient_id": patient_id,
                    "y_true": float(y_test[row_index]),
                    "y_pred": float(y_pred[row_index]),
                    "y_score": float(y_score[row_index]) if y_score is not None else np.nan,
                }
            )
    return metrics_rows, prediction_rows


def build_utility_only_views(source_df: pd.DataFrame, patient_ids: list[str] | None = None) -> pd.DataFrame:
    view = source_df.groupby("patient_id", as_index=False).agg(utility_label=("utility_label", "first"))
    if patient_ids is not None:
        order = pd.DataFrame({"patient_id": patient_ids, "_order": range(len(patient_ids))})
        view = order.merge(view, on="patient_id", how="inner").sort_values("_order").drop(columns="_order")
    view["utility_indicator"] = (view["utility_label"].astype(str) == "Atrial").astype(float)
    return view


def evaluate_constant_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    seed: int,
) -> list[dict]:
    rows = []
    for task in ["sex", "age_group", "age"]:
        train_merged = train_df[["patient_id"]].drop_duplicates().merge(
            demographics_df[["patient_id", "age", "sex"]], on="patient_id", how="inner"
        )
        test_merged = test_df[["patient_id"]].drop_duplicates().merge(
            demographics_df[["patient_id", "age", "sex"]], on="patient_id", how="inner"
        )
        if task == "sex":
            train_merged = train_merged[train_merged["sex"].isin(["Male", "Female"])]
            test_merged = test_merged[test_merged["sex"].isin(["Male", "Female"])]
            y_train = (train_merged["sex"] == "Female").to_numpy(dtype=int)
            y_test = (test_merged["sex"] == "Female").to_numpy(dtype=int)
        elif task == "age_group":
            train_merged = train_merged[train_merged["age"].notna()]
            test_merged = test_merged[test_merged["age"].notna()]
            y_train = (train_merged["age"] >= 65.0).to_numpy(dtype=int)
            y_test = (test_merged["age"] >= 65.0).to_numpy(dtype=int)
        else:
            y_train = np.asarray([])
            y_test = np.asarray([])

        if task in {"sex", "age_group"}:
            prevalence = float(np.mean(y_train))
            prediction = int(prevalence >= 0.5)
            y_pred = np.full(len(y_test), prediction, dtype=int)
            y_score = np.full(len(y_test), prevalence, dtype=float)
        else:
            train_merged = train_merged[train_merged["age"].notna()]
            test_merged = test_merged[test_merged["age"].notna()]
            y_train = train_merged["age"].to_numpy(dtype=float)
            y_test = test_merged["age"].to_numpy(dtype=float)
            y_pred = np.full(len(y_test), np.median(y_train), dtype=float)
            y_score = None
        rows.append(
            {
                "seed": seed,
                "task": task,
                "representation": "constant_baseline",
                "train_condition": "single_segment",
                "test_condition": "single_segment",
                "n_train": int(len(y_train)),
                **evaluate_task_predictions(task, y_test, y_pred, y_score=y_score),
            }
        )
    return rows


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["task", "representation", "train_condition", "test_condition"]
    metric_columns = [
        "n_train",
        "n_test",
        "balanced_accuracy",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "pr_auc",
        "mae_years",
        "rmse_years",
        "r2",
        "spearman_r",
    ]
    available_metrics = [column for column in metric_columns if column in metrics_df]
    summary = metrics_df.groupby(group_columns, as_index=False)[available_metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in column if part]) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def plot_attribute_results(metrics_df: pd.DataFrame, link_df: pd.DataFrame, output_prefix: Path) -> list[str]:
    output_paths = []
    condition_order = ["single_segment", "oracle_group", "link_selected_group", "random_group"]
    representation_order = ["raw_features", "encoder", "encoder_gaussian_noise"]

    for task, metric, ylabel, suffix in [
        ("sex", "balanced_accuracy", "Balanced accuracy", "sex_balanced_accuracy"),
        ("age_group", "balanced_accuracy", "Balanced accuracy", "age_group_balanced_accuracy"),
        ("age", "mae_years", "MAE (years; lower is better)", "age_mae"),
    ]:
        selected = metrics_df[
            (metrics_df["task"] == task)
            & (metrics_df["representation"].isin(representation_order))
        ].copy()
        if selected.empty:
            continue
        means = selected.groupby(["representation", "test_condition"])[metric].mean().unstack()
        means = means.reindex(index=representation_order, columns=condition_order)
        ax = means.plot(kind="bar", figsize=(9, 4.8), width=0.82)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_xticklabels(["Raw features", "Encoder", "Encoder + noise"], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Inference input", fontsize=8)
        plt.tight_layout()
        path = output_prefix.with_name(f"{output_prefix.name}_{suffix}.png")
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        output_paths.append(str(path))

    if not link_df.empty:
        link_means = link_df.groupby("representation")["selection_precision"].mean().reindex(representation_order)
        ax = link_means.plot(kind="bar", figsize=(7, 4.3), color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylabel("Precision of selected linked segments")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(["Raw features", "Encoder", "Encoder + noise"], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = output_prefix.with_name(f"{output_prefix.name}_link_selection_precision.png")
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        output_paths.append(str(path))
    return output_paths


def evaluate_representation(
    representation: str,
    train_representation_df: pd.DataFrame,
    test_representation_df: pd.DataFrame,
    query_ids: list[str],
    demographics_df: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
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
    train_single = build_true_view(train_representation_df, feature_columns, refs=[args.group_refs[0]])
    train_group = build_true_view(train_representation_df, feature_columns, refs=args.group_refs)

    metrics_rows = []
    prediction_rows = []
    for task in ["sex", "age_group", "age"]:
        rows, predictions = fit_and_evaluate_attribute_model(
            task=task,
            train_view=train_single,
            test_views={"single_segment": test_views["single_segment"]},
            demographics_df=demographics_df,
            feature_columns=feature_columns,
            representation=representation,
            train_condition="single_segment",
            seed=seed,
            n_estimators=args.attribute_estimators,
        )
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)

        rows, predictions = fit_and_evaluate_attribute_model(
            task=task,
            train_view=train_group,
            test_views={
                "oracle_group": test_views["oracle_group"],
                "link_selected_group": test_views["link_selected_group"],
                "random_group": test_views["random_group"],
            },
            demographics_df=demographics_df,
            feature_columns=feature_columns,
            representation=representation,
            train_condition="oracle_group",
            seed=seed,
            n_estimators=args.attribute_estimators,
        )
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)

    link_metrics.update(
        {
            "seed": seed,
            "representation": representation,
            "train_pairs": int(len(train_pair_df)),
            "train_pair_positive_prevalence": float(train_pair_df["pair_label"].mean()),
        }
    )
    return metrics_rows, prediction_rows, link_metrics


def main() -> None:
    args = parse_args()
    started = time.time()
    if len(args.group_refs) < 2 or len(set(args.group_refs)) != len(args.group_refs):
        raise ValueError("--group-refs must contain at least two unique segment references.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    features_df["patient_id"] = features_df["patient_id"].astype(str)

    demographics_df, metadata_coverage = load_demographic_metadata(
        records_dir=args.records_dir,
        cache_path=args.metadata_cache,
        required_ids=set(features_df["patient_id"].unique()),
        workers=args.metadata_workers,
    )

    metrics_rows = []
    prediction_rows = []
    link_rows = []
    encoder_protocols = []

    for seed in args.seeds:
        print(f"Seed {seed}: creating record-disjoint train/test split...")
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        query_ids = choose_query_ids(
            test_df,
            demographics_df=demographics_df,
            group_refs=args.group_refs,
            n_queries=args.attack_queries,
            seed=seed,
        )

        metrics_rows.extend(
            evaluate_constant_baselines(
                train_df=train_df,
                test_df=build_utility_only_views(test_df, patient_ids=query_ids),
                demographics_df=demographics_df,
                seed=seed,
            )
        )
        utility_train = build_utility_only_views(train_df)
        utility_test = build_utility_only_views(test_df, patient_ids=query_ids)
        for task in ["sex", "age_group", "age"]:
            rows, predictions = fit_and_evaluate_attribute_model(
                task=task,
                train_view=utility_train,
                test_views={"single_segment": utility_test},
                demographics_df=demographics_df,
                feature_columns=["utility_indicator"],
                representation="utility_only",
                train_condition="single_segment",
                seed=seed,
                n_estimators=args.attribute_estimators,
            )
            metrics_rows.extend(rows)
            prediction_rows.extend(predictions)

        print(f"Seed {seed}: evaluating raw feature representation...")
        rows, predictions, link_metrics = evaluate_representation(
            representation="raw_features",
            train_representation_df=train_df,
            test_representation_df=test_df,
            query_ids=query_ids,
            demographics_df=demographics_df,
            seed=seed,
            args=args,
        )
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)
        link_rows.append(link_metrics)

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

        print(f"Seed {seed}: evaluating encoder representation...")
        rows, predictions, link_metrics = evaluate_representation(
            representation="encoder",
            train_representation_df=train_encoder_df,
            test_representation_df=test_encoder_df,
            query_ids=query_ids,
            demographics_df=demographics_df,
            seed=seed,
            args=args,
        )
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)
        link_rows.append(link_metrics)

        train_noisy_df, test_noisy_df, noise_metadata = apply_dp_to_embedding_split(
            train_embedding_df=train_encoder_df,
            test_embedding_df=test_encoder_df,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            clip_norm=args.dp_clip_norm,
            seed=seed + 30_000,
        )
        print(f"Seed {seed}: evaluating Gaussian-noised encoder representation...")
        rows, predictions, link_metrics = evaluate_representation(
            representation="encoder_gaussian_noise",
            train_representation_df=train_noisy_df,
            test_representation_df=test_noisy_df,
            query_ids=query_ids,
            demographics_df=demographics_df,
            seed=seed,
            args=args,
        )
        link_metrics.update(noise_metadata)
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)
        link_rows.append(link_metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    link_df = pd.DataFrame(link_rows)
    summary_df = summarize_metrics(metrics_df)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    predictions_path = args.output_dir / f"{args.run_name}_predictions.csv.gz"
    link_path = args.output_dir / f"{args.run_name}_link_selection.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False, compression="gzip")
    link_df.to_csv(link_path, index=False)

    figure_prefix = args.figure_dir / args.run_name
    figure_paths = plot_attribute_results(metrics_df, link_df, output_prefix=figure_prefix)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "records_dir": str(args.records_dir),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "manifest_total_segments": manifest.get("total_segments"),
        "loaded_segments": int(len(features_df)),
        "loaded_identifiers": int(features_df["patient_id"].nunique()),
        "metadata_coverage": metadata_coverage,
        "seeds": args.seeds,
        "test_size": args.test_size,
        "split_unit": "retained record identifier",
        "group_refs": args.group_refs,
        "group_size": len(args.group_refs),
        "aggregation": "feature-wise mean",
        "attack_queries": args.attack_queries,
        "gallery_negatives": args.gallery_negatives,
        "linkability_attacker": {
            "model": "XGBoost on absolute representation differences",
            "train_max_pairs": args.train_max_pairs,
            "min_segment_gap": args.min_segment_gap,
            "max_positive_pairs_per_identifier": args.max_positive_pairs_per_patient,
            "selection": (
                "For each anchor, rank group_size-1 true same-record candidates mixed with gallery negatives "
                "and aggregate the top group_size-1 candidates with the anchor."
            ),
        },
        "attribute_attacker": {
            "sex": "XGBoost binary classifier; Female is the positive class",
            "age_group": "XGBoost binary classifier; age >= 65 years is the positive class",
            "age": "XGBoost regressor",
            "estimators": args.attribute_estimators,
            "training_views": (
                "single-segment models are trained on one anchor per record; grouped models are trained on "
                "oracle same-record aggregates and tested on oracle, link-selected, and random aggregates"
            ),
        },
        "representations": ["raw_features", "encoder", "encoder_gaussian_noise"],
        "raw_representation_note": "raw_features denotes the 208 handcrafted segment features, not ECG samples",
        "encoder": {
            "bottleneck_dim": args.bottleneck_dim,
            "hidden_units": args.encoder_hidden_units,
            "max_iter": args.encoder_max_iter,
            "train_max_segments": args.encoder_train_max_segments,
            "per_seed": encoder_protocols,
        },
        "gaussian_noise": {
            "epsilon": args.dp_epsilon,
            "delta": args.dp_delta,
            "clip_norm": args.dp_clip_norm,
            "scope_note": (
                "The experiment reuses the article's clipped standardized embedding plus Gaussian noise. "
                "Results must not be described as end-to-end differential privacy."
            ),
        },
        "claim_boundary": (
            "The retained identifier denotes one ECG record and is not independently validated as a "
            "longitudinal person identity. The experiment measures same-record aggregation and attribute inference."
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "predictions": str(predictions_path),
            "link_selection": str(link_path),
            "figures": figure_paths,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved link selection metrics:", link_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
