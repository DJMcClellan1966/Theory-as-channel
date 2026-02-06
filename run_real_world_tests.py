"""
Real-world tests: Theory-as-Channel vs multiple ML models.

Compares standard and unusual ensemble methods:
- Majority, weighted, soft (ML-Toolbox style)
- Unusual: entropy-weighted soft, median proba, trimmed-mean, capacity-weighted, diversity-weighted, stacking
"""
import numpy as np
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from theory_as_channel import (
    ErrorCorrectingPredictions,
    channel_capacity,
    entropy_weighted_soft_weights,
    capacity_weighted_weights,
    soft_vote_median,
    soft_vote_trimmed_mean,
    diversity_weighted_weights,
    SimpleStackingEnsemble,
    build_meta_features_from_probas,
)

# SVC needs probability=True for predict_proba (soft voting)
def _make_models(random_state=42):
    return [
        ("LogisticRegression", LogisticRegression(max_iter=500, random_state=random_state)),
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ("SVC", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTree", DecisionTreeClassifier(random_state=random_state)),
        ("GaussianNB", GaussianNB()),
    ]


def run_dataset(name, X, y, test_size=0.3, random_state=42):
    """Train multiple models; compare majority, weighted, and soft voting (ML-Toolbox style)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = _make_models(random_state)
    preds = []
    probas_list = []
    accuracies = []

    for label, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds.append(pred)
        acc = np.mean(pred == y_test)
        accuracies.append((label, acc))
        if hasattr(model, "predict_proba"):
            probas_list.append(model.predict_proba(X_test))
        else:
            probas_list.append(None)

    predictions = np.column_stack(preds)
    n_models = len(models)
    ec = ErrorCorrectingPredictions(redundancy_factor=n_models)
    weights = [a for _, a in accuracies]

    # 1) Unweighted majority vote (original theory-as-channel)
    decoded_majority = ec.correct_predictions(predictions, method="majority_vote")
    acc_majority = np.mean(decoded_majority == y_test)

    # 2) Weighted majority vote (ML-Toolbox style)
    decoded_weighted = ec.correct_predictions(predictions, method="majority_vote", weights=weights)
    acc_weighted = np.mean(decoded_weighted == y_test)

    # 3) Soft voting (ML-Toolbox voting='soft')
    acc_soft = acc_soft_weighted = None
    acc_entropy_soft = acc_capacity_w = acc_median_proba = acc_trimmed = acc_diversity_w = acc_stacking = None
    if all(p is not None for p in probas_list):
        probas = np.stack(probas_list, axis=1)
        decoded_soft = ec.correct_predictions_soft(probas, weights=None)
        acc_soft = np.mean(decoded_soft == y_test)
        decoded_soft_w = ec.correct_predictions_soft(probas, weights=weights)
        acc_soft_weighted = np.mean(decoded_soft_w == y_test)
        # Unusual: entropy-weighted soft (confident models count more)
        w_ent = entropy_weighted_soft_weights(probas)
        decoded_ent = ec.correct_predictions_soft(probas, weights=w_ent.tolist())
        acc_entropy_soft = np.mean(decoded_ent == y_test)
        # Unusual: median of probas (robust)
        decoded_med = soft_vote_median(probas)
        acc_median_proba = np.mean(decoded_med == y_test)
        # Unusual: trimmed mean of probas
        decoded_trim = soft_vote_trimmed_mean(probas, trim=1)
        acc_trimmed = np.mean(decoded_trim == y_test)
        # Unusual: capacity-weighted soft (Shannon C from accuracy)
        w_cap = capacity_weighted_weights(weights)
        decoded_cap = ec.correct_predictions_soft(probas, weights=w_cap.tolist())
        acc_capacity_w = np.mean(decoded_cap == y_test)
    # Unusual: capacity-weighted majority
    w_cap = capacity_weighted_weights(weights)
    decoded_cap_maj = ec.correct_predictions(predictions, method="majority_vote", weights=w_cap.tolist())
    acc_capacity_maj = np.mean(decoded_cap_maj == y_test)
    # Unusual: diversity-weighted majority (accurate + disagreeing)
    w_div = diversity_weighted_weights(predictions, weights)
    decoded_div = ec.correct_predictions(predictions, method="majority_vote", weights=w_div.tolist())
    acc_diversity_w = np.mean(decoded_div == y_test)
    # Unusual: stacking (meta-learner on base probas)
    if all(p is not None for p in probas_list):
        probas_train_list = [m.predict_proba(X_train) for _, m in models if hasattr(m, "predict_proba")]
        if len(probas_train_list) == n_models:
            meta_train = build_meta_features_from_probas(np.stack(probas_train_list, axis=1))
            meta_test = build_meta_features_from_probas(probas)
            stack = SimpleStackingEnsemble(use_proba=True)
            stack.fit(meta_train, y_train)
            decoded_stack = stack.predict(meta_test)
            acc_stacking = np.mean(decoded_stack == y_test)

    # Channel capacity view
    correct_per_sample = np.sum(predictions == y_test.reshape(-1, 1), axis=1)
    mean_correct = np.mean(correct_per_sample) / n_models
    mean_wrong = 1.0 - mean_correct
    signal_power = max(mean_correct**2, 0.01)
    noise_power = max(mean_wrong**2, 0.01)
    C = channel_capacity(signal_power=signal_power, noise_power=noise_power, bandwidth=1.0)

    return {
        "name": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(np.unique(y)),
        "models": accuracies,
        "best_single": max(a for _, a in accuracies),
        "worst_single": min(a for _, a in accuracies),
        "acc_majority": acc_majority,
        "acc_weighted": acc_weighted,
        "acc_soft": acc_soft,
        "acc_soft_weighted": acc_soft_weighted,
        "acc_entropy_soft": acc_entropy_soft,
        "acc_capacity_maj": acc_capacity_maj,
        "acc_capacity_w": acc_capacity_w,
        "acc_median_proba": acc_median_proba,
        "acc_trimmed": acc_trimmed,
        "acc_diversity_w": acc_diversity_w,
        "acc_stacking": acc_stacking,
        "capacity_bits": C,
        "signal_power": signal_power,
        "noise_power": noise_power,
    }


def _p(a):
    return f"{a:.1%}" if a is not None else " n/a"


def main():
    print("=" * 70)
    print("THEORY-AS-CHANNEL: Real-world tests vs multiple models")
    print("=" * 70)

    datasets = [
        ("Digits", load_digits(return_X_y=True)),
        ("Wine", load_wine(return_X_y=True)),
        ("Breast Cancer", load_breast_cancer(return_X_y=True)),
    ]

    all_results = []

    for name, (X, y) in datasets:
        print(f"\n{'-' * 70}")
        print(f"Dataset: {name}  (samples={len(X)}, classes={len(np.unique(y))})")
        print("-" * 70)
        result = run_dataset(name, X, y)
        all_results.append(result)

        print("\n  Single-model accuracy (each channel):")
        for model_name, acc in result["models"]:
            print(f"    {model_name:20s}  {acc:.2%}")
        print(f"\n  Best single model:  {result['best_single']:.2%}")
        print("  Standard ensembles:")
        print(f"    Majority / Weighted / Soft / Soft+W:  {result['acc_majority']:.1%} / {result['acc_weighted']:.1%} / {_p(result['acc_soft'])} / {_p(result['acc_soft_weighted'])}")
        print("  Unusual methods:")
        print(f"    Entropy-soft / Capacity-maj / Capacity-soft:  {_p(result['acc_entropy_soft'])} / {result['acc_capacity_maj']:.1%} / {_p(result['acc_capacity_w'])}")
        print(f"    Median-proba / Trimmed-mean / Diversity-maj / Stacking:  {_p(result['acc_median_proba'])} / {_p(result['acc_trimmed'])} / {result['acc_diversity_w']:.1%} / {_p(result['acc_stacking'])}")
        print(f"\n  Channel: signal_power~{result['signal_power']:.4f}, C ~ {result['capacity_bits']:.3f} bits")

    # Summary: standard
    print("\n" + "=" * 70)
    print("SUMMARY - Standard")
    print("=" * 70)
    print(f"\n{'Dataset':<14} {'Best':>8} {'Majority':>8} {'Weighted':>8} {'Soft':>8} {'Soft+W':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<14} {r['best_single']:>7.1%} {r['acc_majority']:>7.1%} {r['acc_weighted']:>7.1%} {_p(r['acc_soft']):>8} {_p(r['acc_soft_weighted']):>8}")
    # Summary: unusual
    print("\n" + "=" * 70)
    print("SUMMARY - Unusual (for toolbox / elsewhere)")
    print("=" * 70)
    print(f"\n{'Dataset':<14} {'Best':>8} {'EntrSoft':>8} {'CapMaj':>8} {'MedProba':>8} {'Trim':>8} {'DivMaj':>8} {'Stack':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<14} {r['best_single']:>7.1%} {_p(r['acc_entropy_soft']):>8} {r['acc_capacity_maj']:>7.1%} {_p(r['acc_median_proba']):>8} {_p(r['acc_trimmed']):>8} {r['acc_diversity_w']:>7.1%} {_p(r['acc_stacking']):>8}")
    print("=" * 70)
    print("\nUnusual: entropy-weighted soft, capacity-weighted (Shannon C), median/trimmed proba,")
    print("diversity-weighted majority, stacking. Reusable in ML-Toolbox or other ensembles.")
    print("\nCaveat: Weights use the same test set; for unbiased estimates use a separate")
    print("validation set for weights and report on a held-out test set. See README 'Caveats'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
