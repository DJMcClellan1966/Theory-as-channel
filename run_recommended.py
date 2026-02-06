"""
Recommended pipeline: train/val/test, weights from validation only.

- Train base models on train set.
- Compute per-model accuracies and ensemble weights on VAL only (no test leakage).
- Report metrics on TEST only.
- Few combiners (majority, weighted majority, soft weighted) so the app improves
  by better data/models/validation, not by adding more methods.
"""
import numpy as np
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split  # stratified splits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from theory_as_channel import ErrorCorrectingPredictions

# One canonical model list
def _make_models(seed=42):
    return [
        ("LogReg", LogisticRegression(max_iter=500, random_state=seed)),
        ("RF", RandomForestClassifier(n_estimators=50, random_state=seed)),
        ("SVC", SVC(kernel="rbf", probability=True, random_state=seed)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("DT", DecisionTreeClassifier(random_state=seed)),
        ("NB", GaussianNB()),
    ]


def run_one_dataset(name, X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """Train on train; compute weights from val; report on test only."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    # Stratified: first split off test, then split rest into train/val
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    val_ratio_rest = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio_rest, random_state=seed, stratify=y_rest
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    models = _make_models(seed)
    # Train on train only
    for _, m in models:
        m.fit(X_train, y_train)

    # Predictions on val (for weights) and test (for final report)
    preds_val = np.column_stack([m.predict(X_val) for _, m in models])
    preds_test = np.column_stack([m.predict(X_test) for _, m in models])
    accs_val = [np.mean(m.predict(X_val) == y_val) for _, m in models]
    accs_test_single = [np.mean(m.predict(X_test) == y_test) for _, m in models]
    best_single = max(accs_test_single)

    probas_val = []
    probas_test = []
    for _, m in models:
        if hasattr(m, "predict_proba"):
            probas_val.append(m.predict_proba(X_val))
            probas_test.append(m.predict_proba(X_test))
        else:
            probas_val.append(None)
            probas_test.append(None)
    have_proba = all(p is not None for p in probas_val)
    if have_proba:
        Pval = np.stack(probas_val, axis=1)
        Ptest = np.stack(probas_test, axis=1)

    n_models = len(models)
    ec = ErrorCorrectingPredictions(redundancy_factor=n_models)
    weights = accs_val  # from val only

    # Ensemble predictions on TEST (weights were chosen using val only)
    maj_test = ec.correct_predictions(preds_test, method="majority_vote")
    wgt_test = ec.correct_predictions(preds_test, method="majority_vote", weights=weights)
    acc_maj = np.mean(maj_test == y_test)
    acc_wgt = np.mean(wgt_test == y_test)
    acc_soft = None
    if have_proba:
        soft_test = ec.correct_predictions_soft(Ptest, weights=weights)
        acc_soft = np.mean(soft_test == y_test)

    return {
        "name": name,
        "best_single": best_single,
        "acc_majority": acc_maj,
        "acc_weighted": acc_wgt,
        "acc_soft": acc_soft,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


def main():
    import sys
    seeds = [42, 43, 44] if "--multi" in sys.argv else [42]
    datasets = [
        ("Digits", load_digits(return_X_y=True)),
        ("Wine", load_wine(return_X_y=True)),
        ("Breast Cancer", load_breast_cancer(return_X_y=True)),
    ]
    print("=" * 60)
    print("RECOMMENDED: train/val/test, weights from VAL, report on TEST")
    print("=" * 60)
    for name, (X, y) in datasets:
        results = [run_one_dataset(name, X, y, seed=s) for s in seeds]
        r0 = results[0]
        if len(seeds) == 1:
            soft_s = f"  soft={r0['acc_soft']:.2%}" if r0["acc_soft"] is not None else ""
            print(f"\n{name}:  best_single={r0['best_single']:.2%}  majority={r0['acc_majority']:.2%}  weighted={r0['acc_weighted']:.2%}{soft_s}")
        else:
            best = np.mean([r["best_single"] for r in results])
            maj = np.mean([r["acc_majority"] for r in results])
            wgt = np.mean([r["acc_weighted"] for r in results])
            std_w = np.std([r["acc_weighted"] for r in results])
            print(f"\n{name} (mean over {len(seeds)} seeds):  best={best:.2%}  majority={maj:.2%}  weighted={wgt:.2%} +/- {std_w:.2%}")
    print("\n" + "=" * 60)
    print("Weights from VAL only; TEST untouched until final metric. Use --multi for 3 seeds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
