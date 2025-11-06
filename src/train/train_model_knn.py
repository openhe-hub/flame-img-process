"""
Train a K-Nearest Neighbors regressor on the flame dataset.
"""

from pathlib import Path
import sys

from loguru import logger
from sklearn.neighbors import KNeighborsRegressor

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from train_common import evaluate_regression_model, prepare_dataset  # noqa: E402


def train_flame_model():
    """Train and evaluate a KNN regressor."""
    logger.info("Starting KNeighborsRegressor training workflow...")
    splits = prepare_dataset()

    model = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
    model.fit(splits.X_train, splits.y_train)
    logger.success("KNeighborsRegressor training completed.")

    evaluate_regression_model(model, splits, model_name="KNeighborsRegressor")


if __name__ == "__main__":
    train_flame_model()
