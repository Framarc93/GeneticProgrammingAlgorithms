import numpy as np
import pytest

from examples.regression.evaluate_functions import evaluate_regression, evaluate_regression_noVal


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf, 1e308])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("invalid_split", ["train", "validation", "both"])
def test_invalid_rmse_with_validation(prediction, scalar, invalid_split):
    train = np.array([[0.0, 0.0]])
    validation = np.array([[1.0, 1.0]])

    def predict(x):
        invalid = invalid_split == "both" or (
            invalid_split == "train" if x[0] == 0 else invalid_split == "validation"
        )
        value = prediction if invalid else 0.0
        return value if scalar else np.full(x.shape, value)

    with np.errstate(over="ignore", invalid="ignore"):
        result = evaluate_regression(
            None,
            lambda _: predict,
            kwargs={
                "X_train": train,
                "y_train": np.zeros(2),
                "X_val": validation,
                "y_val": np.zeros(2),
            },
        )
    assert result == (
        1e10 if invalid_split in ("train", "both") else 0.0,
        0.0,
        1e10 if invalid_split in ("validation", "both") else 0.0,
        0.0,
    )


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf, 1e308, 2e10])
@pytest.mark.parametrize("scalar", [False, True])
def test_rmse_without_validation(prediction, scalar):
    def predict(x):
        return prediction if scalar else np.full(x.shape, prediction)

    with np.errstate(over="ignore", invalid="ignore"):
        result = evaluate_regression_noVal(
            None,
            lambda _: predict,
            kwargs={"X_train": np.zeros((1, 2)), "y_train": np.zeros(2)},
        )
    # Finite scores are preserved, even when larger than the invalid-score fallback.
    expected = 2e10 if prediction == 2e10 else 1e10
    assert result == (expected, 0.0)
