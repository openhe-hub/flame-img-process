"""
Train a Random Forest model for interpolation: hold out a small case (default experiment_name contains 'Ar-10') for testing.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple

DATA_CSV_PATH = Path("data/dataset/total_dataset.csv")
TARGET_VARIABLE = "area"

LABEL_COLUMNS = [
    "area",
    "arc_length",
    "area_vec",
    "regression_circle_center",
    "regression_circle_radius",
    "expand_dist",
    "expand_vec",
    "tip_distance",
]

FEATURES_TO_DROP = ["experiment_name", "frame_id"] + LABEL_COLUMNS
HOLDOUT_COLUMN = "experiment_name"
HOLDOUT_VALUES = {"Ar-10"}
HOLDOUT_MODE = "contains"  # options: 'exact', 'contains', 'regex'


def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        logger.success(f"Loaded dataset from '{DATA_CSV_PATH}' with shape {df.shape}")
    except FileNotFoundError as exc:
        logger.error(
            f"Dataset not found at '{DATA_CSV_PATH}'. Please generate it before training."
        )
        raise exc

    before = len(df)
    df.drop_duplicates(subset=["experiment_name", "frame_id"], keep="first", inplace=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} duplicate rows. New shape: {df.shape}")
    else:
        logger.info("No duplicates detected.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def _build_holdout_mask(values: pd.Series) -> pd.Series:
    series = values.astype(str)
    targets = [str(v) for v in HOLDOUT_VALUES]

    if HOLDOUT_MODE == "exact":
        return series.isin(targets)
    if HOLDOUT_MODE == "contains":
        pattern = "|".join(re.escape(target) for target in targets)
        return series.str.contains(pattern, na=False)
    if HOLDOUT_MODE == "regex":
        pattern = "|".join(targets)
        return series.str.contains(pattern, na=False, regex=True)

    raise ValueError(
        f"Unsupported HOLDOUT_MODE '{HOLDOUT_MODE}'. Choose from 'exact', 'contains', 'regex'."
    )


def split_interpolation_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if HOLDOUT_COLUMN not in df.columns:
        raise ValueError(
            f"Hold-out column '{HOLDOUT_COLUMN}' not found in dataset columns: {list(df.columns)}"
        )

    mask = _build_holdout_mask(df[HOLDOUT_COLUMN])
    test_df = df[mask].copy()
    train_df = df[~mask].copy()

    holdout_desc = ", ".join(sorted(str(v) for v in HOLDOUT_VALUES))
    logger.info(
        f"Interpolation split: {len(train_df)} training samples, {len(test_df)} testing samples "
        f"({HOLDOUT_COLUMN} {HOLDOUT_MODE} {{{holdout_desc}}})."
    )

    if test_df.empty:
        raise ValueError(
            f"No samples found with {HOLDOUT_COLUMN} {HOLDOUT_MODE} {sorted(HOLDOUT_VALUES)}. "
            "Interpolation evaluation cannot proceed."
        )

    return train_df, test_df


def train_model() -> None:
    holdout_desc = ", ".join(sorted(str(v) for v in HOLDOUT_VALUES))
    logger.info(
        "Starting Random Forest interpolation experiment "
        f"(hold out rows where {HOLDOUT_COLUMN} {HOLDOUT_MODE} {{{holdout_desc}}})..."
    )

    df = load_dataset()
    train_df, test_df = split_interpolation_sets(df)

    y_train = train_df[TARGET_VARIABLE]
    X_train = train_df.drop(columns=FEATURES_TO_DROP, errors="ignore")

    y_test = test_df[TARGET_VARIABLE]
    X_test = test_df.drop(columns=FEATURES_TO_DROP, errors="ignore")

    X_train = pd.get_dummies(X_train, columns=["gas_type"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["gas_type"], drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)

    feature_names = X_train.columns

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    logger.info("Training RandomForestRegressor on interpolation training split...")
    model.fit(X_train_scaled, y_train)
    logger.success("Training completed.")

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Interpolation Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"Interpolation R-squared (R²) Score: {r2:.6f}")

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(min(10, len(importances)))
    logger.info("Top feature importances for interpolation experiment:")
    print(top_features)

    print("10 Error Example: \n", y_test[:10] - y_pred[:10])


if __name__ == "__main__":
    train_model()
