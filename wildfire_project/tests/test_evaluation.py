import numpy as np
import pytest

from wildfire_ml.evaluation import calculate_binary_metrics, select_fbeta_threshold


def test_requested_binary_metrics_have_expected_values() -> None:
    result = calculate_binary_metrics(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5, beta=2
    )

    assert result.metrics["accuracy_score"] == pytest.approx(0.75)
    assert result.metrics["precision_score"] == pytest.approx(1.0)
    assert result.metrics["recall_score"] == pytest.approx(0.5)
    assert result.metrics["f1_score"] == pytest.approx(2 / 3)
    assert result.metrics["fbeta_score"] == pytest.approx(5 / 9)
    assert result.metrics["roc_auc_score"] == pytest.approx(0.75)
    assert result.metrics["auc"] == pytest.approx(0.75)
    assert result.metrics["average_precision_score"] == pytest.approx(5 / 6)
    assert result.metrics["brier_score_loss"] == pytest.approx(
        np.mean((np.array([0.1, 0.4, 0.35, 0.8]) - np.array([0, 0, 1, 1])) ** 2)
    )
    assert result.metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert len(result.curves["roc_curve.false_positive_rate"]) > 0
    assert len(result.curves["precision_recall_curve.precision"]) > 0


def test_threshold_is_selected_from_validation_predictions() -> None:
    threshold, score = select_fbeta_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], beta=2
    )

    assert threshold == pytest.approx(0.35)
    assert score == pytest.approx(10 / 11)


@pytest.mark.parametrize("probability", [[-0.1, 0.2], [0.2, np.nan], [0.2]])
def test_invalid_probabilities_are_rejected(probability: list[float]) -> None:
    with pytest.raises(ValueError):
        calculate_binary_metrics([0, 1], probability)

