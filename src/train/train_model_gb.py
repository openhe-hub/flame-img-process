"""
Train a Gradient Boosting Regressor on the flame dataset.
"""

from pathlib import Path
import sys

from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from train_common import evaluate_regression_model, prepare_dataset  # noqa: E402


def train_flame_model():
    """Train and evaluate a Gradient Boosting regressor."""
    logger.info("Starting GradientBoostingRegressor training workflow...")
    splits = prepare_dataset()

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(splits.X_train, splits.y_train)
    logger.success("GradientBoostingRegressor training completed.")

    evaluate_regression_model(model, splits, model_name="GradientBoostingRegressor")


if __name__ == "__main__":
    train_flame_model()
