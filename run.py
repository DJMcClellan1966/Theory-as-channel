"""Theory-as-Channel viability test."""
import numpy as np

from theory_as_channel import (
    ErrorCorrectingPredictions,
    channel_capacity,
)


def run():
    # 1) Error correction: 3 models, one occasionally wrong -> majority vote should fix
    ec = ErrorCorrectingPredictions(redundancy_factor=3)
    np.random.seed(42)
    n = 100
    true_labels = np.random.randint(0, 2, n)
    # Simulate 3 models: model0 correct, model1 80% correct, model2 80% correct
    pred0 = true_labels.copy()
    pred1 = np.where(np.random.rand(n) < 0.8, true_labels, 1 - true_labels)
    pred2 = np.where(np.random.rand(n) < 0.8, true_labels, 1 - true_labels)
    predictions = np.column_stack([pred0, pred1, pred2])
    corrected = ec.correct_predictions(predictions, method="majority_vote")
    acc_raw = np.mean(pred1 == true_labels)  # single model
    acc_corrected = np.mean(corrected == true_labels)
    # 2) Channel capacity
    C = channel_capacity(signal_power=10.0, noise_power=1.0, bandwidth=1.0)
    assert C > 0, "channel_capacity should be positive"
    return {
        "ok": True,
        "single_model_acc": float(acc_raw),
        "corrected_acc": float(acc_corrected),
        "channel_capacity_bits": float(C),
        "correction_helps": acc_corrected >= acc_raw,
    }


if __name__ == "__main__":
    import sys
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
