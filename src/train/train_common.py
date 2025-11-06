"""
Shared utilities for training flame regression models with scikit-learn.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_CSV_PATH = Path("data/dataset/total_dataset.csv")
LABEL_COLUMNS: Sequence[str] = (
    "area",
    "arc_length",
    "area_vec",
    "regression_circle_center",
    "regression_circle_radius",
    "expand_dist",
    "expand_vec",
    "tip_distance",
)
DEFAULT_FEATURES_TO_DROP: Sequence[str] = ("experiment_name", "frame_id")
DEFAULT_CATEGORICAL_COLUMNS: Sequence[str] = ("gas_type",)


@dataclass
class DatasetSplits:
    """Container for train/test splits and metadata."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: pd.Index
    scaler: StandardScaler


def prepare_dataset(
    target_variable: str = "area",
    *,
    data_path: Path = DATA_CSV_PATH,
    features_to_drop: Optional[Iterable[str]] = None,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """
    Load the aggregated dataset, perform standard preprocessing, and return train/test splits.
    """

    logger.info("Starting dataset preparation...")
    logger.debug(f"Target variable: '{target_variable}'")
    logger.debug(f"Reading dataset from '{data_path}'")

    try:
        df = pd.read_csv(data_path)
        logger.success(f"Loaded dataset with shape {df.shape}")
    except FileNotFoundError as exc:
        logger.exception(
            f"Dataset not found at {data_path}. Please verify the path before training."
        )
        raise exc

    # Deduplicate based on experiment and frame identifiers
    initial_rows = len(df)
    df.drop_duplicates(subset=["experiment_name", "frame_id"], keep="first", inplace=True)
    if len(df) < initial_rows:
        logger.warning(f"Removed {initial_rows - len(df)} duplicate rows. New shape: {df.shape}")
    else:
        logger.info("No duplicate rows detected.")

    # Handle missing values for numerical columns
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Separate features and target
    labels = LABEL_COLUMNS
    drop_columns = list(DEFAULT_FEATURES_TO_DROP)
    if features_to_drop:
        drop_columns.extend(features_to_drop)
    drop_columns.extend(labels)

    if target_variable not in labels:
        logger.warning(
            f"Target variable '{target_variable}' is not in label list. "
            "Ensure FEATURES_TO_DROP includes the correct columns."
        )

    y = df[target_variable]
    X = df.drop(columns=drop_columns, errors="ignore")

    logger.info("Feature columns after dropping reserved columns:")
    logger.info(list(X.columns))

    # Apply one-hot encoding to categorical columns that are present
    columns_to_encode = [col for col in categorical_columns if col in X.columns]
    if columns_to_encode:
        X = pd.get_dummies(X, columns=columns_to_encode, drop_first=True)
        logger.info(f"Applied one-hot encoding to columns: {columns_to_encode}")
    else:
        logger.info("No categorical columns found for encoding.")

    feature_names = X.columns
    logger.info(f"{len(feature_names)} features will be used for training.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples."
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Applied standard scaling to numeric features.")

    return DatasetSplits(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def evaluate_regression_model(
    model,
    splits: DatasetSplits,
    *,
    model_name: Optional[str] = None,
    sample_size: int = 10,
    random_state: int = 42,
) -> None:
    """
    Evaluate the trained model, emitting key metrics and diagnostics via the logger.
    """

    model_label = model_name or model.__class__.__name__
    logger.info(f"Evaluating {model_label} on the test set...")

    y_pred = model.predict(splits.X_test)
    mse = mean_squared_error(splits.y_test, y_pred)
    r2 = r2_score(splits.y_test, y_pred)

    logger.info(f"{model_label} Mean Squared Error: {mse:.4f}")
    logger.info(f"{model_label} R-squared: {r2:.4f}")

    # Feature importance or coefficients, if available
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=splits.feature_names)
        top_features = importances.nlargest(min(10, len(importances)))
        logger.info(f"{model_label} top features by importance:")
        print(top_features)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if isinstance(coef, np.ndarray):
            flat_coef = coef.reshape(-1)
            coefficients = pd.Series(flat_coef, index=splits.feature_names[: len(flat_coef)])
            top_features = coefficients.abs().nlargest(min(10, len(coefficients)))
            logger.info(f"{model_label} top features by absolute coefficient:")
            print(coefficients[top_features.index])
    else:
        logger.info(f"{model_label} does not expose feature importances or coefficients.")

    # Display prediction samples
    results_df = pd.DataFrame(
        {
            "Ground Truth": splits.y_test,
            "Predicted Value": y_pred,
        }
    )
    results_df["Absolute Error"] = (results_df["Ground Truth"] - results_df["Predicted Value"]).abs()

    n_samples = min(sample_size, len(results_df))
    if n_samples > 0:
        logger.info(f"Displaying {n_samples} sample predictions for {model_label}:")
        print(results_df.sample(n_samples, random_state=random_state))
