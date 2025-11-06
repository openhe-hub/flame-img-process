"""
Train a Support Vector Regressor on the flame dataset.
"""

from pathlib import Path
import sys

from loguru import logger
from sklearn.svm import SVR

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from train_common import evaluate_regression_model, prepare_dataset  # noqa: E402


def train_flame_model():
    """Train and evaluate an SVR model."""
    logger.info("Starting Support Vector Regressor training workflow...")
    splits = prepare_dataset()

    model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    model.fit(splits.X_train, splits.y_train)
    logger.success("SVR training completed.")

    evaluate_regression_model(model, splits, model_name="SupportVectorRegressor")


if __name__ == "__main__":
    train_flame_model()
