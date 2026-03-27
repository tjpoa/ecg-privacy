from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


METADATA_COLUMNS = {
    "patient_id",
    "record_id",
    "segment_id",
    "label",
    "utility_label",
    "segment_ref",
    "start_sample",
    "end_sample",
}


def resolve_positive_label(label_encoder, preferred_labels=None):
    classes = [str(label) for label in label_encoder.classes_]

    if preferred_labels is None:
        preferred_labels = ["Atrial", "AFL"]

    for label in preferred_labels:
        if label in classes:
            return int(label_encoder.transform([label])[0])

    # Fallback: use the first class in encoder order, which is typically the
    # minority/positive class when labels are explicitly named.
    return 0


def get_binary_targets(label_encoder, series, preferred_labels=None):
    encoded = label_encoder.transform(series)
    positive_encoded = resolve_positive_label(label_encoder, preferred_labels=preferred_labels)
    binary_targets = (encoded == positive_encoded).astype(int)

    positive_name = str(label_encoder.inverse_transform([positive_encoded])[0])
    negative_names = [str(label) for idx, label in enumerate(label_encoder.classes_) if idx != positive_encoded]
    negative_name = negative_names[0] if negative_names else "Negative"

    return binary_targets, positive_name, negative_name


def get_feature_columns(features_df, exclude_columns=None):
    exclude = set(METADATA_COLUMNS)
    if exclude_columns is not None:
        exclude.update(exclude_columns)

    return [col for col in features_df.columns if col not in exclude]


def split_segments_by_patient(features_df, group_col="patient_id", test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(features_df, groups=features_df[group_col]))
    train_df = features_df.iloc[train_idx].reset_index(drop=True)
    test_df = features_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def split_segments_by_patient_stratified(
    features_df,
    group_col="patient_id",
    label_col="utility_label",
    test_size=0.2,
    random_state=42,
):
    patient_df = features_df[[group_col, label_col]].drop_duplicates().reset_index(drop=True)
    label_counts = patient_df[label_col].value_counts()
    if len(label_counts) < 2 or label_counts.min() < 2:
        return split_segments_by_patient(
            features_df,
            group_col=group_col,
            test_size=test_size,
            random_state=random_state,
        )

    train_patients, test_patients = train_test_split(
        patient_df,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_df[label_col],
    )
    train_patient_ids = set(train_patients[group_col])
    test_patient_ids = set(test_patients[group_col])

    train_df = features_df[features_df[group_col].isin(train_patient_ids)].reset_index(drop=True)
    test_df = features_df[features_df[group_col].isin(test_patient_ids)].reset_index(drop=True)
    return train_df, test_df


def split_train_val_test_by_patient(
    features_df,
    group_col="patient_id",
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    label_col=None,
    stratified=False,
):
    split_fn = split_segments_by_patient_stratified if stratified else split_segments_by_patient
    split_kwargs = {
        "group_col": group_col,
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratified and label_col is not None:
        split_kwargs["label_col"] = label_col
    train_val_df, test_df = split_fn(features_df, **split_kwargs)
    relative_val_size = val_size / (1.0 - test_size)
    split_kwargs.update({"test_size": relative_val_size, "random_state": random_state + 1})
    train_df, val_df = split_fn(train_val_df, **split_kwargs)
    return train_df, val_df, test_df


def aggregate_scores_by_patient(eval_df, score_col="score", group_col="patient_id", label_col="utility_label"):
    return (
        eval_df[[group_col, label_col, score_col]]
        .groupby(group_col, as_index=False)
        .agg({label_col: "first", score_col: "mean"})
    )


def _build_logistic_pipeline(feature_columns):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def _build_imputer():
    return SimpleImputer(strategy="median")


def _build_random_forest_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )


def _build_gradient_boosting_model():
    return GradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
    )


def normalize_requested_models(models):
    if models is None:
        return ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost"]
    return list(models)


def _build_xgb_model(**overrides):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed in the current environment.")

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def _build_xgb_model_with_balance(scale_pos_weight, **overrides):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed in the current environment.")

    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def get_pair_feature_names(feature_columns, representation="absdiff"):
    if representation == "absdiff":
        return [f"absdiff__{col}" for col in feature_columns]
    if representation == "absdiff_mean":
        return [f"absdiff__{col}" for col in feature_columns] + [f"mean__{col}" for col in feature_columns]
    raise ValueError("representation must be 'absdiff' or 'absdiff_mean'.")


def evaluate_binary_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    positive_label=1,
    target_names=None,
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if classes and positive_label in classes:
            positive_index = classes.index(positive_label)
        else:
            positive_index = 1
        y_score = model.predict_proba(X_test)[:, positive_index]
    else:
        y_score = model.decision_function(X_test)

    results = {
        "model": model_name,
        "f1_score": f1_score(y_test, y_pred, pos_label=positive_label),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            target_names=target_names,
        ),
    }
    return results


def get_positive_class_scores(model, X, positive_label=1):
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if classes and positive_label in classes:
            positive_index = classes.index(positive_label)
        else:
            positive_index = 1
        return model.predict_proba(X)[:, positive_index]
    return model.decision_function(X)


def find_best_threshold(y_true, y_score, positive_label=1, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, pos_label=positive_label)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_binary_scores(
    y_true,
    y_score,
    model_name,
    positive_label=1,
    target_names=None,
    threshold=0.5,
):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": threshold,
        "f1_score": f1_score(y_true, y_pred, pos_label=positive_label),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            target_names=target_names,
        ),
    }


def run_utility_baselines(
    features_df,
    target_col="utility_label",
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    stratified=True,
    models=None,
):
    split_fn = split_segments_by_patient_stratified if stratified else split_segments_by_patient
    split_kwargs = {
        "group_col": group_col,
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratified:
        split_kwargs["label_col"] = target_col
    train_df, test_df = split_fn(features_df, **split_kwargs)

    feature_columns = get_feature_columns(features_df)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    y_train, positive_name, negative_name = get_binary_targets(label_encoder, train_df[target_col])
    y_test, _, _ = get_binary_targets(label_encoder, test_df[target_col])
    positive_label = 1
    target_names = [negative_name, positive_name]
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    imputer = _build_imputer().fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    requested_models = normalize_requested_models(models)
    results = []

    if "LogisticRegression" in requested_models:
        logistic_pipeline = _build_logistic_pipeline(feature_columns)
        results.append(
            evaluate_binary_classifier(
                logistic_pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name="LogisticRegression",
                positive_label=positive_label,
                target_names=target_names,
            )
        )

    if "RandomForest" in requested_models:
        random_forest_model = _build_random_forest_model()
        results.append(
            evaluate_binary_classifier(
                random_forest_model,
                X_train_imputed,
                y_train,
                X_test_imputed,
                y_test,
                model_name="RandomForest",
                positive_label=positive_label,
                target_names=target_names,
            )
        )

    if "GradientBoosting" in requested_models:
        gradient_boosting_model = _build_gradient_boosting_model()
        results.append(
            evaluate_binary_classifier(
                gradient_boosting_model,
                X_train_imputed,
                y_train,
                X_test_imputed,
                y_test,
                model_name="GradientBoosting",
                positive_label=positive_label,
                target_names=target_names,
            )
        )

    if "XGBoost" in requested_models and XGBClassifier is not None:
        xgb_model = _build_xgb_model()
        results.append(
            evaluate_binary_classifier(
                xgb_model,
                X_train_imputed,
                y_train,
                X_test_imputed,
                y_test,
                model_name="XGBoost",
                positive_label=positive_label,
                target_names=target_names,
            )
        )

    summary_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"confusion_matrix", "classification_report"}} for result in results]
    )
    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
        "results": results,
        "summary_df": summary_df,
    }


def evaluate_utility_model_on_split(
    train_df,
    test_df,
    model_name="LogisticRegression",
    target_col="utility_label",
    feature_columns=None,
):
    if feature_columns is None:
        feature_columns = get_feature_columns(train_df)
    else:
        feature_columns = list(feature_columns)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    y_train, positive_name, negative_name = get_binary_targets(label_encoder, train_df[target_col])
    y_test, _, _ = get_binary_targets(label_encoder, test_df[target_col])
    positive_label = 1
    target_names = [negative_name, positive_name]
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    imputer = _build_imputer().fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    if model_name == "LogisticRegression":
        model = _build_logistic_pipeline(feature_columns)
        evaluation = evaluate_binary_classifier(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )
    elif model_name == "RandomForest":
        model = _build_random_forest_model()
        evaluation = evaluate_binary_classifier(
            model,
            X_train_imputed,
            y_train,
            X_test_imputed,
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )
    elif model_name == "GradientBoosting":
        model = _build_gradient_boosting_model()
        evaluation = evaluate_binary_classifier(
            model,
            X_train_imputed,
            y_train,
            X_test_imputed,
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )
    elif model_name == "XGBoost":
        model = _build_xgb_model()
        evaluation = evaluate_binary_classifier(
            model,
            X_train_imputed,
            y_train,
            X_test_imputed,
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )
    else:
        raise ValueError(
            "model_name must be 'LogisticRegression', 'RandomForest', 'GradientBoosting', or 'XGBoost'."
        )

    return {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "evaluation": evaluation,
    }


def run_utility_baselines_tuned(
    features_df,
    target_col="utility_label",
    group_col="patient_id",
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    threshold_grid=None,
    xgb_params=None,
    stratified=True,
    models=None,
):
    train_df, val_df, test_df = split_train_val_test_by_patient(
        features_df,
        group_col=group_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        label_col=target_col,
        stratified=stratified,
    )

    feature_columns = get_feature_columns(features_df)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    y_train, positive_name, negative_name = get_binary_targets(label_encoder, train_df[target_col])
    y_val, _, _ = get_binary_targets(label_encoder, val_df[target_col])
    y_test, _, _ = get_binary_targets(label_encoder, test_df[target_col])
    positive_label = 1
    target_names = [negative_name, positive_name]

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]
    imputer = _build_imputer().fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    combined_train = pd.concat([train_df, val_df], ignore_index=True)
    X_train_full = combined_train[feature_columns]
    imputer_full = _build_imputer().fit(X_train_full)
    X_train_full_imputed = imputer_full.transform(X_train_full)
    y_train_full, _, _ = get_binary_targets(label_encoder, combined_train[target_col])

    requested_models = normalize_requested_models(models)
    results = []

    if "LogisticRegression" in requested_models:
        logistic_model = _build_logistic_pipeline(feature_columns)
        logistic_model.fit(X_train, y_train)
        val_scores = get_positive_class_scores(logistic_model, X_val, positive_label=positive_label)
        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_scores,
            positive_label=positive_label,
            thresholds=threshold_grid,
        )

        logistic_model_final = _build_logistic_pipeline(feature_columns)
        logistic_model_final.fit(X_train_full, y_train_full)
        test_scores = get_positive_class_scores(logistic_model_final, X_test, positive_label=positive_label)
        logistic_results = evaluate_binary_scores(
            y_test,
            test_scores,
            model_name="LogisticRegression_tuned",
            positive_label=positive_label,
            target_names=target_names,
            threshold=best_threshold,
        )
        logistic_results["val_f1_at_best_threshold"] = best_val_f1
        results.append(logistic_results)

    if "RandomForest" in requested_models:
        random_forest_model = _build_random_forest_model()
        random_forest_model.fit(X_train_imputed, y_train)
        val_scores = get_positive_class_scores(random_forest_model, X_val_imputed, positive_label=positive_label)
        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_scores,
            positive_label=positive_label,
            thresholds=threshold_grid,
        )
        random_forest_model_final = _build_random_forest_model()
        random_forest_model_final.fit(X_train_full_imputed, y_train_full)
        test_scores = get_positive_class_scores(
            random_forest_model_final,
            X_test_imputed,
            positive_label=positive_label,
        )
        random_forest_results = evaluate_binary_scores(
            y_test,
            test_scores,
            model_name="RandomForest_tuned",
            positive_label=positive_label,
            target_names=target_names,
            threshold=best_threshold,
        )
        random_forest_results["val_f1_at_best_threshold"] = best_val_f1
        results.append(random_forest_results)

    if "GradientBoosting" in requested_models:
        gradient_boosting_model = _build_gradient_boosting_model()
        gradient_boosting_model.fit(X_train_imputed, y_train)
        val_scores = get_positive_class_scores(gradient_boosting_model, X_val_imputed, positive_label=positive_label)
        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_scores,
            positive_label=positive_label,
            thresholds=threshold_grid,
        )
        gradient_boosting_model_final = _build_gradient_boosting_model()
        gradient_boosting_model_final.fit(X_train_full_imputed, y_train_full)
        test_scores = get_positive_class_scores(
            gradient_boosting_model_final,
            X_test_imputed,
            positive_label=positive_label,
        )
        gradient_boosting_results = evaluate_binary_scores(
            y_test,
            test_scores,
            model_name="GradientBoosting_tuned",
            positive_label=positive_label,
            target_names=target_names,
            threshold=best_threshold,
        )
        gradient_boosting_results["val_f1_at_best_threshold"] = best_val_f1
        results.append(gradient_boosting_results)

    if "XGBoost" in requested_models and XGBClassifier is not None:
        positives = max(1, int(np.sum(y_train == positive_label)))
        negatives = max(1, int(np.sum(y_train != positive_label)))
        scale_pos_weight = negatives / positives

        xgb_model = _build_xgb_model_with_balance(scale_pos_weight=scale_pos_weight, **(xgb_params or {}))
        xgb_model.fit(X_train_imputed, y_train)
        val_scores = get_positive_class_scores(xgb_model, X_val_imputed, positive_label=positive_label)
        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_scores,
            positive_label=positive_label,
            thresholds=threshold_grid,
        )

        positives_full = max(1, int(np.sum(y_train_full == positive_label)))
        negatives_full = max(1, int(np.sum(y_train_full != positive_label)))
        scale_pos_weight_full = negatives_full / positives_full
        xgb_model_final = _build_xgb_model_with_balance(
            scale_pos_weight=scale_pos_weight_full,
            **(xgb_params or {}),
        )
        xgb_model_final.fit(X_train_full_imputed, y_train_full)
        test_scores = get_positive_class_scores(xgb_model_final, X_test_imputed, positive_label=positive_label)
        xgb_results = evaluate_binary_scores(
            y_test,
            test_scores,
            model_name="XGBoost_tuned",
            positive_label=positive_label,
            target_names=target_names,
            threshold=best_threshold,
        )
        xgb_results["val_f1_at_best_threshold"] = best_val_f1
        results.append(xgb_results)

    summary_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"confusion_matrix", "classification_report"}} for result in results]
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
        "results": results,
        "summary_df": summary_df,
    }


def fit_utility_model(
    features_df,
    model_name="LogisticRegression",
    target_col="utility_label",
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    stratified=True,
    feature_columns=None,
):
    split_fn = split_segments_by_patient_stratified if stratified else split_segments_by_patient
    split_kwargs = {
        "group_col": group_col,
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratified:
        split_kwargs["label_col"] = target_col
    train_df, test_df = split_fn(features_df, **split_kwargs)
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)
    else:
        feature_columns = list(feature_columns)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    y_train, positive_name, negative_name = get_binary_targets(label_encoder, train_df[target_col])
    y_test, _, _ = get_binary_targets(label_encoder, test_df[target_col])
    positive_label = 1
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    if model_name == "LogisticRegression":
        model = _build_logistic_pipeline(feature_columns)
        model.fit(X_train, y_train)
    elif model_name == "RandomForest":
        imputer = _build_imputer().fit(X_train)
        model = Pipeline(
            steps=[
                ("imputer", imputer),
                ("model", _build_random_forest_model()),
            ]
        )
        model.fit(X_train, y_train)
    elif model_name == "GradientBoosting":
        imputer = _build_imputer().fit(X_train)
        model = Pipeline(
            steps=[
                ("imputer", imputer),
                ("model", _build_gradient_boosting_model()),
            ]
        )
        model.fit(X_train, y_train)
    elif model_name == "XGBoost":
        imputer = _build_imputer().fit(X_train)
        model = Pipeline(
            steps=[
                ("imputer", imputer),
                ("model", _build_xgb_model()),
            ]
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(
            "model_name must be 'LogisticRegression', 'RandomForest', 'GradientBoosting', or 'XGBoost'."
        )

    return {
        "model_name": model_name,
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
        "positive_label": positive_label,
        "target_names": [negative_name, positive_name],
    }


def evaluate_utility_feature_subset(
    features_df,
    feature_columns,
    model_name="LogisticRegression",
    target_col="utility_label",
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    stratified=True,
):
    fitted = fit_utility_model(
        features_df=features_df,
        model_name=model_name,
        target_col=target_col,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
        stratified=stratified,
        feature_columns=feature_columns,
    )
    model = fitted["model"]
    X_train = fitted["X_train"]
    X_test = fitted["X_test"]
    y_train = fitted["y_train"]
    y_test = fitted["y_test"]
    positive_label = fitted["positive_label"]
    target_names = fitted["target_names"]

    if model_name == "LogisticRegression":
        evaluation = evaluate_binary_classifier(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )
    else:
        evaluation = evaluate_binary_classifier(
            model,
            X_train.to_numpy(),
            y_train,
            X_test.to_numpy(),
            y_test,
            model_name=model_name,
            positive_label=positive_label,
            target_names=target_names,
        )

    evaluation["n_features"] = len(feature_columns)
    evaluation["feature_columns"] = list(feature_columns)
    return {
        "fitted": fitted,
        "evaluation": evaluation,
    }


def run_utility_feature_subset_experiments(
    features_df,
    ranked_features,
    subset_sizes,
    model_name="LogisticRegression",
    target_col="utility_label",
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    stratified=True,
):
    available_features = set(get_feature_columns(features_df))
    ranked_features = [feature for feature in ranked_features if feature in available_features]

    results = []
    for subset_size in subset_sizes:
        selected_features = ranked_features[:subset_size]
        if not selected_features:
            continue

        experiment = evaluate_utility_feature_subset(
            features_df=features_df,
            feature_columns=selected_features,
            model_name=model_name,
            target_col=target_col,
            group_col=group_col,
            test_size=test_size,
            random_state=random_state,
            stratified=stratified,
        )
        evaluation = experiment["evaluation"]
        results.append(
            {
                "model": evaluation["model"],
                "subset_size": subset_size,
                "f1_score": evaluation["f1_score"],
                "f1_macro": evaluation["f1_macro"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "roc_auc": evaluation["roc_auc"],
                "pr_auc": evaluation["pr_auc"],
                "feature_columns": selected_features,
            }
        )

    return pd.DataFrame(results)


def generate_positive_pairs(
    segment_df,
    max_pairs=None,
    random_state=42,
    min_segment_gap=0,
    max_pairs_per_patient=None,
):
    rng = np.random.default_rng(random_state)
    positive_pairs = []

    for _, group in segment_df.groupby("patient_id"):
        group = group.sort_values("segment_ref").reset_index()
        records = group[["index", "segment_ref"]].to_dict("records")
        patient_pairs = []

        for left, right in combinations(records, 2):
            if abs(int(left["segment_ref"]) - int(right["segment_ref"])) < min_segment_gap:
                continue
            patient_pairs.append((int(left["index"]), int(right["index"])))

        if max_pairs_per_patient is not None and len(patient_pairs) > max_pairs_per_patient:
            selected = rng.choice(len(patient_pairs), size=max_pairs_per_patient, replace=False)
            patient_pairs = [patient_pairs[idx] for idx in selected]

        positive_pairs.extend(patient_pairs)

    if max_pairs is not None and len(positive_pairs) > max_pairs:
        selected = rng.choice(len(positive_pairs), size=max_pairs, replace=False)
        positive_pairs = [positive_pairs[idx] for idx in selected]

    return positive_pairs


def generate_negative_pairs(segment_df, n_pairs, random_state=42, max_attempts_factor=20):
    rng = np.random.default_rng(random_state)
    negative_pairs = []
    patient_ids = segment_df["patient_id"].to_numpy()
    indices = segment_df.index.to_numpy()
    pair_set = set()
    max_attempts = max(n_pairs * max_attempts_factor, 1000)
    attempts = 0

    while len(negative_pairs) < n_pairs and attempts < max_attempts:
        idx_a, idx_b = rng.choice(indices, size=2, replace=False)
        attempts += 1

        if patient_ids[idx_a] == patient_ids[idx_b]:
            continue

        pair = tuple(sorted((int(idx_a), int(idx_b))))
        if pair in pair_set:
            continue

        pair_set.add(pair)
        negative_pairs.append(pair)

    return negative_pairs


def build_pair_table(
    segment_df,
    max_pairs=None,
    random_state=42,
    min_segment_gap=0,
    max_pairs_per_patient=None,
):
    segment_df = segment_df.reset_index(drop=True)
    positive_pairs = generate_positive_pairs(
        segment_df,
        max_pairs=max_pairs,
        random_state=random_state,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_pairs_per_patient,
    )
    negative_pairs = generate_negative_pairs(
        segment_df,
        n_pairs=len(positive_pairs),
        random_state=random_state,
    )

    rows = []
    for idx_a, idx_b in positive_pairs:
        rows.append({"idx_a": idx_a, "idx_b": idx_b, "pair_label": 1})
    for idx_a, idx_b in negative_pairs:
        rows.append({"idx_a": idx_a, "idx_b": idx_b, "pair_label": 0})

    return pd.DataFrame(rows)


def build_pair_features(features_df, pair_df, feature_columns, representation="absdiff"):
    X_rows = []

    for _, row in pair_df.iterrows():
        xa = features_df.loc[row["idx_a"], feature_columns].to_numpy(dtype=float)
        xb = features_df.loc[row["idx_b"], feature_columns].to_numpy(dtype=float)
        absdiff = np.abs(xa - xb)

        if representation == "absdiff":
            pair_features = absdiff
        elif representation == "absdiff_mean":
            pair_features = np.concatenate([absdiff, (xa + xb) / 2.0])
        else:
            raise ValueError("representation must be 'absdiff' or 'absdiff_mean'.")

        X_rows.append(pair_features)

    X_pair = np.vstack(X_rows) if X_rows else np.empty((0, 0), dtype=float)
    y_pair = pair_df["pair_label"].to_numpy(dtype=int)
    return X_pair, y_pair


def _prepare_standardized_feature_matrices(train_df, test_df, feature_columns):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_matrix = imputer.fit_transform(train_df[feature_columns])
    test_matrix = imputer.transform(test_df[feature_columns])

    train_matrix = scaler.fit_transform(train_matrix)
    test_matrix = scaler.transform(test_matrix)
    return train_matrix, test_matrix


def _compute_pair_similarity_scores(feature_matrix, pair_df, metric="euclidean"):
    scores = []
    eps = 1e-8

    for _, row in pair_df.iterrows():
        xa = feature_matrix[int(row["idx_a"])]
        xb = feature_matrix[int(row["idx_b"])]

        if metric == "euclidean":
            score = -float(np.linalg.norm(xa - xb))
        elif metric == "cosine":
            denom = (np.linalg.norm(xa) * np.linalg.norm(xb)) + eps
            score = float(np.dot(xa, xb) / denom)
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'.")

        scores.append(score)

    return np.asarray(scores, dtype=float)


def evaluate_distance_linkability_baseline(
    train_df,
    test_df,
    train_pair_df,
    test_pair_df,
    feature_columns,
    metric="euclidean",
):
    train_matrix, test_matrix = _prepare_standardized_feature_matrices(train_df, test_df, feature_columns)
    y_train = train_pair_df["pair_label"].to_numpy(dtype=int)
    y_test = test_pair_df["pair_label"].to_numpy(dtype=int)

    train_scores = _compute_pair_similarity_scores(train_matrix, train_pair_df, metric=metric)
    test_scores = _compute_pair_similarity_scores(test_matrix, test_pair_df, metric=metric)
    threshold, _ = find_best_threshold(y_train, train_scores, positive_label=1)

    model_name = "EuclideanDistance" if metric == "euclidean" else "CosineSimilarity"
    return evaluate_binary_scores(
        y_true=y_test,
        y_score=test_scores,
        model_name=model_name,
        positive_label=1,
        target_names=["different_patient", "same_patient"],
        threshold=threshold,
    )


def run_linkability_baselines(
    features_df,
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    max_pairs=5000,
    representation="absdiff",
    min_segment_gap=0,
    max_positive_pairs_per_patient=None,
    include_distance_baselines=False,
):
    train_df, test_df = split_segments_by_patient(
        features_df,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )

    feature_columns = get_feature_columns(features_df)
    train_pair_df = build_pair_table(
        train_df,
        max_pairs=max_pairs,
        random_state=random_state,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )
    test_pair_df = build_pair_table(
        test_df,
        max_pairs=max_pairs,
        random_state=random_state + 1,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )

    X_train, y_train = build_pair_features(train_df, train_pair_df, feature_columns, representation=representation)
    X_test, y_test = build_pair_features(test_df, test_pair_df, feature_columns, representation=representation)

    results = []

    if include_distance_baselines:
        results.append(
            evaluate_distance_linkability_baseline(
                train_df=train_df,
                test_df=test_df,
                train_pair_df=train_pair_df,
                test_pair_df=test_pair_df,
                feature_columns=feature_columns,
                metric="euclidean",
            )
        )
        results.append(
            evaluate_distance_linkability_baseline(
                train_df=train_df,
                test_df=test_df,
                train_pair_df=train_pair_df,
                test_pair_df=test_pair_df,
                feature_columns=feature_columns,
                metric="cosine",
            )
        )

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    results.append(
        evaluate_binary_classifier(
            logistic_model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name="LogisticRegression",
            positive_label=1,
            target_names=["different_patient", "same_patient"],
        )
    )

    if XGBClassifier is not None:
        xgb_model = _build_xgb_model()
        results.append(
            evaluate_binary_classifier(
                xgb_model,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name="XGBoost",
                positive_label=1,
                target_names=["different_patient", "same_patient"],
            )
        )

    summary_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"confusion_matrix", "classification_report"}} for result in results]
    )
    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "train_pair_df": train_pair_df,
        "test_pair_df": test_pair_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "summary_df": summary_df,
    }


def run_linkability_baselines_on_split(
    train_df,
    test_df,
    max_pairs=5000,
    representation="absdiff",
    random_state=42,
    min_segment_gap=0,
    max_positive_pairs_per_patient=None,
    include_distance_baselines=False,
):
    feature_columns = get_feature_columns(train_df)
    train_pair_df = build_pair_table(
        train_df,
        max_pairs=max_pairs,
        random_state=random_state,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )
    test_pair_df = build_pair_table(
        test_df,
        max_pairs=max_pairs,
        random_state=random_state + 1,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )

    X_train, y_train = build_pair_features(train_df, train_pair_df, feature_columns, representation=representation)
    X_test, y_test = build_pair_features(test_df, test_pair_df, feature_columns, representation=representation)

    results = []

    if include_distance_baselines:
        results.append(
            evaluate_distance_linkability_baseline(
                train_df=train_df,
                test_df=test_df,
                train_pair_df=train_pair_df,
                test_pair_df=test_pair_df,
                feature_columns=feature_columns,
                metric="euclidean",
            )
        )
        results.append(
            evaluate_distance_linkability_baseline(
                train_df=train_df,
                test_df=test_df,
                train_pair_df=train_pair_df,
                test_pair_df=test_pair_df,
                feature_columns=feature_columns,
                metric="cosine",
            )
        )

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    results.append(
        evaluate_binary_classifier(
            logistic_model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name="LogisticRegression",
            positive_label=1,
            target_names=["different_patient", "same_patient"],
        )
    )

    if XGBClassifier is not None:
        xgb_model = _build_xgb_model()
        results.append(
            evaluate_binary_classifier(
                xgb_model,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name="XGBoost",
                positive_label=1,
                target_names=["different_patient", "same_patient"],
            )
        )

    summary_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"confusion_matrix", "classification_report"}} for result in results]
    )
    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "train_pair_df": train_pair_df,
        "test_pair_df": test_pair_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "summary_df": summary_df,
    }


def fit_linkability_model(
    features_df,
    model_name="LogisticRegression",
    group_col="patient_id",
    test_size=0.2,
    random_state=42,
    max_pairs=5000,
    representation="absdiff",
    min_segment_gap=0,
    max_positive_pairs_per_patient=None,
):
    train_df, test_df = split_segments_by_patient(
        features_df,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )
    feature_columns = get_feature_columns(features_df)
    train_pair_df = build_pair_table(
        train_df,
        max_pairs=max_pairs,
        random_state=random_state,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )
    test_pair_df = build_pair_table(
        test_df,
        max_pairs=max_pairs,
        random_state=random_state + 1,
        min_segment_gap=min_segment_gap,
        max_pairs_per_patient=max_positive_pairs_per_patient,
    )
    X_train, y_train = build_pair_features(train_df, train_pair_df, feature_columns, representation=representation)
    X_test, y_test = build_pair_features(test_df, test_pair_df, feature_columns, representation=representation)

    if model_name == "LogisticRegression":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        )
        model.fit(X_train, y_train)
    elif model_name == "XGBoost":
        model = _build_xgb_model()
        model.fit(X_train, y_train)
    else:
        raise ValueError("model_name must be 'LogisticRegression' or 'XGBoost'.")

    return {
        "model_name": model_name,
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "train_pair_df": train_pair_df,
        "test_pair_df": test_pair_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "pair_feature_names": get_pair_feature_names(feature_columns, representation=representation),
        "representation": representation,
    }


def run_repeated_linkability_baselines(
    features_df,
    random_states,
    group_col="patient_id",
    test_size=0.2,
    max_pairs=5000,
    representation="absdiff",
    min_segment_gap=0,
    max_positive_pairs_per_patient=None,
    include_distance_baselines=False,
):
    detailed_results = []

    for random_state in random_states:
        run_output = run_linkability_baselines(
            features_df=features_df,
            group_col=group_col,
            test_size=test_size,
            random_state=random_state,
            max_pairs=max_pairs,
            representation=representation,
            min_segment_gap=min_segment_gap,
            max_positive_pairs_per_patient=max_positive_pairs_per_patient,
            include_distance_baselines=include_distance_baselines,
        )
        seed_df = run_output["summary_df"].copy()
        seed_df["random_state"] = random_state
        detailed_results.append(seed_df)

    detailed_df = pd.concat(detailed_results, ignore_index=True)
    metric_columns = ["f1_score", "f1_macro", "balanced_accuracy", "roc_auc", "pr_auc"]
    summary_df = detailed_df.groupby("model")[metric_columns].agg(["mean", "std"]).reset_index()
    flattened_columns = []
    for column in summary_df.columns.tolist():
        if isinstance(column, tuple):
            left, right = column
            if left in {"model", "index"}:
                flattened_columns.append("model")
            elif right:
                flattened_columns.append(f"{left}_{right}")
            else:
                flattened_columns.append(str(left))
        else:
            flattened_columns.append(str(column))
    summary_df.columns = flattened_columns

    return {
        "detailed_df": detailed_df,
        "summary_df": summary_df,
    }


def _extract_model_based_importance(model, feature_names):
    if isinstance(model, Pipeline):
        final_model = model.named_steps["model"]
        if hasattr(final_model, "coef_"):
            values = np.abs(final_model.coef_).ravel()
            return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
                "importance", ascending=False
            )
        model = final_model

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
            "importance", ascending=False
        )

    raise ValueError("Model does not expose coefficients or feature_importances_.")


def _compute_permutation_importance(model, X_test, y_test, feature_names, scoring="roc_auc", random_state=42):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring=scoring,
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def compute_utility_feature_importance(
    features_df,
    model_name="LogisticRegression",
    test_size=0.2,
    random_state=42,
    permutation_scoring="average_precision",
):
    fitted = fit_utility_model(
        features_df,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
    )
    model = fitted["model"]
    X_test = fitted["X_test"]
    y_test = fitted["y_test"]
    feature_columns = fitted["feature_columns"]

    model_based_df = _extract_model_based_importance(model, feature_columns)
    permutation_df = _compute_permutation_importance(
        model,
        X_test if model_name == "LogisticRegression" else X_test.to_numpy(),
        y_test,
        feature_columns,
        scoring=permutation_scoring,
        random_state=random_state,
    )

    return {
        "fitted": fitted,
        "model_based_df": model_based_df.reset_index(drop=True),
        "permutation_df": permutation_df.reset_index(drop=True),
    }


def compute_linkability_feature_importance(
    features_df,
    model_name="LogisticRegression",
    test_size=0.2,
    random_state=42,
    max_pairs=5000,
    representation="absdiff",
    min_segment_gap=0,
    max_positive_pairs_per_patient=None,
    permutation_scoring="roc_auc",
):
    fitted = fit_linkability_model(
        features_df,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        max_pairs=max_pairs,
        representation=representation,
        min_segment_gap=min_segment_gap,
        max_positive_pairs_per_patient=max_positive_pairs_per_patient,
    )
    model = fitted["model"]
    X_test = fitted["X_test"]
    y_test = fitted["y_test"]
    pair_feature_names = fitted["pair_feature_names"]

    model_based_df = _extract_model_based_importance(model, pair_feature_names)
    permutation_df = _compute_permutation_importance(
        model,
        X_test,
        y_test,
        pair_feature_names,
        scoring=permutation_scoring,
        random_state=random_state,
    )

    return {
        "fitted": fitted,
        "model_based_df": model_based_df.reset_index(drop=True),
        "permutation_df": permutation_df.reset_index(drop=True),
    }
