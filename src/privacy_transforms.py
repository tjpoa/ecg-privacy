from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    from modeling import METADATA_COLUMNS, get_feature_columns
except ImportError:  # pragma: no cover - package import fallback
    from .modeling import METADATA_COLUMNS, get_feature_columns


@dataclass
class FittedFeatureTransform:
    name: str
    feature_columns: list[str]
    output_feature_names: list[str]
    imputer: SimpleImputer | None = None
    scaler: object | None = None
    pca: PCA | None = None
    decimals: int | None = None
    noise_std: float | None = None
    random_state: int | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    projector: GaussianRandomProjection | None = None


def _copy_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [col for col in df.columns if col in METADATA_COLUMNS]
    return df[metadata_columns].copy()


def _build_output_df(metadata_df: pd.DataFrame, array: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    transformed_df = pd.DataFrame(array, columns=feature_names, index=metadata_df.index)
    return pd.concat([metadata_df.reset_index(drop=True), transformed_df.reset_index(drop=True)], axis=1)


def fit_feature_transform(
    train_df: pd.DataFrame,
    method: str,
    feature_columns: list[str] | None = None,
    **kwargs,
) -> FittedFeatureTransform:
    if feature_columns is None:
        feature_columns = get_feature_columns(train_df)
    else:
        feature_columns = list(feature_columns)

    method = method.lower()

    if method == "identity":
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
        )

    imputer = SimpleImputer(strategy="median")
    train_matrix = imputer.fit_transform(train_df[feature_columns])

    if method == "standard":
        scaler = StandardScaler()
        scaler.fit(train_matrix)
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            scaler=scaler,
        )

    if method == "robust":
        scaler = RobustScaler()
        scaler.fit(train_matrix)
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            scaler=scaler,
        )

    if method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(train_matrix)
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            scaler=scaler,
        )

    if method == "quantization":
        decimals = int(kwargs.get("decimals", 2))
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            decimals=decimals,
        )

    if method == "winsorization":
        lower_q = float(kwargs.get("lower_quantile", 0.01))
        upper_q = float(kwargs.get("upper_quantile", 0.99))
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
        lower_bounds = np.quantile(train_matrix, lower_q, axis=0)
        upper_bounds = np.quantile(train_matrix, upper_q, axis=0)
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    if method == "noise":
        noise_std = float(kwargs.get("noise_std", 0.05))
        random_state = int(kwargs.get("random_state", 42))
        scaler = StandardScaler()
        scaler.fit(train_matrix)
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=feature_columns,
            imputer=imputer,
            scaler=scaler,
            noise_std=noise_std,
            random_state=random_state,
        )

    if method == "pca":
        n_components = kwargs.get("n_components", 50)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_matrix)
        pca = PCA(n_components=n_components, random_state=kwargs.get("random_state", 42))
        pca.fit(train_scaled)
        output_feature_names = [f"pca_{idx:03d}" for idx in range(pca.n_components_)]
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=output_feature_names,
            imputer=imputer,
            scaler=scaler,
            pca=pca,
        )

    if method == "random_projection":
        n_components = kwargs.get("n_components", 50)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_matrix)
        projector = GaussianRandomProjection(
            n_components=n_components,
            random_state=kwargs.get("random_state", 42),
        )
        projector.fit(train_scaled)
        output_feature_names = [f"rp_{idx:03d}" for idx in range(int(n_components))]
        return FittedFeatureTransform(
            name=method,
            feature_columns=feature_columns,
            output_feature_names=output_feature_names,
            imputer=imputer,
            scaler=scaler,
            projector=projector,
        )

    raise ValueError(
        "method must be one of: identity, standard, robust, minmax, quantization, winsorization, noise, pca, random_projection."
    )


def transform_feature_dataframe(df: pd.DataFrame, fitted: FittedFeatureTransform, split_name: str = "test") -> pd.DataFrame:
    metadata_df = _copy_metadata(df)

    if fitted.name == "identity":
        return pd.concat([metadata_df.reset_index(drop=True), df[fitted.feature_columns].reset_index(drop=True)], axis=1)

    matrix = fitted.imputer.transform(df[fitted.feature_columns])

    if fitted.name in {"standard", "robust", "minmax"}:
        transformed = fitted.scaler.transform(matrix)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    if fitted.name == "quantization":
        transformed = np.round(matrix, decimals=fitted.decimals)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    if fitted.name == "winsorization":
        transformed = np.clip(matrix, fitted.lower_bounds, fitted.upper_bounds)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    if fitted.name == "noise":
        transformed = fitted.scaler.transform(matrix)
        offset = 0 if split_name == "train" else 10_000
        rng = np.random.default_rng((fitted.random_state or 42) + offset)
        transformed = transformed + rng.normal(loc=0.0, scale=fitted.noise_std, size=transformed.shape)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    if fitted.name == "pca":
        transformed = fitted.scaler.transform(matrix)
        transformed = fitted.pca.transform(transformed)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    if fitted.name == "random_projection":
        transformed = fitted.scaler.transform(matrix)
        transformed = fitted.projector.transform(transformed)
        return _build_output_df(metadata_df, transformed, fitted.output_feature_names)

    raise ValueError(f"Unsupported fitted transform: {fitted.name}")


def fit_transform_feature_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    feature_columns: list[str] | None = None,
    **kwargs,
):
    fitted = fit_feature_transform(
        train_df=train_df,
        method=method,
        feature_columns=feature_columns,
        **kwargs,
    )
    train_transformed = transform_feature_dataframe(train_df, fitted, split_name="train")
    test_transformed = transform_feature_dataframe(test_df, fitted, split_name="test")
    return train_transformed, test_transformed, fitted
