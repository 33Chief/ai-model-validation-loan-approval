import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from fairlearn.metrics import (
    MetricFrame, selection_rate, demographic_parity_difference,
    demographic_parity_ratio, equalized_odds_difference, true_positive_rate,
    false_positive_rate,
)

CHART = "/home/claude/ai_model_validation_project/charts/"
OUT = "/home/claude/ai_model_validation_project/outputs/"

pipe = joblib.load(OUT + "model.joblib")
features = json.load(open(OUT + "features.json"))
test = pd.read_csv(OUT + "test_set.csv")
train = pd.read_csv(OUT + "train_set.csv")

X_test = test[features]
y_test = test["loan_approved"]
proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

results = {}

# ---------- 1. Performance ----------
results["performance"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "brier_score": round(brier_score_loss(y_test, proba), 4),
    "n_test": int(len(y_test)),
    "base_rate": round(float(y_test.mean()), 4),
}

cm = confusion_matrix(y_test, pred)
results["confusion_matrix"] = cm.tolist()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color="#185FA5", linewidth=2, label=f"Model (AUC={results['performance']['roc_auc']})")
plt.plot([0, 1], [0, 1], "--", color="#888780", linewidth=1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve - holdout test set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(CHART + "roc_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(5, 4.5))
cm_pct = cm / cm.sum()
im = plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["Predicted: Decline", "Predicted: Approve"])
plt.yticks([0, 1], ["Actual: Decline", "Actual: Approve"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1%})", ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
plt.title("Confusion matrix - holdout test set")
plt.tight_layout()
plt.savefig(CHART + "confusion_matrix.png", dpi=150)
plt.close()

# ---------- 2. Calibration ----------
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
plt.figure(figsize=(5, 5))
plt.plot(mean_pred, frac_pos, "o-", color="#3B6D11", label="Model")
plt.plot([0, 1], [0, 1], "--", color="#888780", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed approval rate")
plt.title("Calibration curve (10 quantile bins)")
plt.legend()
plt.tight_layout()
plt.savefig(CHART + "calibration.png", dpi=150)
plt.close()

# ---------- 3. Feature importance ----------
importances = pipe.named_steps["clf"].feature_importances_
imp_series = pd.Series(importances, index=features).sort_values()
results["feature_importance"] = imp_series.round(4).to_dict()

plt.figure(figsize=(6, 4))
plt.barh(imp_series.index, imp_series.values, color="#534AB7")
plt.title("Feature importance (Gini-based)")
plt.tight_layout()
plt.savefig(CHART + "feature_importance.png", dpi=150)
plt.close()

# ---------- 4. Stability / drift (simulate a later scoring population) ----------
rng = np.random.default_rng(99)
drift = test.copy()
drift["credit_score"] = np.clip(drift["credit_score"] - rng.normal(14, 4, len(drift)), 300, 850)
drift["annual_income"] = drift["annual_income"] * rng.normal(0.94, 0.03, len(drift))
drift["debt_to_income"] = np.clip(drift["debt_to_income"] * rng.normal(1.08, 0.03, len(drift)), 0.02, 0.9)
drift_proba = pipe.predict_proba(drift[features])[:, 1]

def psi(expected, actual, bins=10):
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    e_counts = np.clip(e_counts, 1e-4, None)
    a_counts = np.clip(a_counts, 1e-4, None)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))

score_psi = psi(proba, drift_proba)
feature_psi = {f: round(psi(test[f].fillna(test[f].median()), drift[f].fillna(drift[f].median())), 4)
               for f in features}

results["stability"] = {
    "score_psi": round(score_psi, 4),
    "feature_psi": feature_psi,
    "simulated_period_approval_rate": round(float((drift_proba >= 0.5).mean()), 4),
    "baseline_approval_rate": round(float((proba >= 0.5).mean()), 4),
}

plt.figure(figsize=(6, 4))
psi_sorted = pd.Series(feature_psi).sort_values()
colors = ["#A32D2D" if v > 0.2 else ("#854F0B" if v > 0.1 else "#3B6D11") for v in psi_sorted.values]
plt.barh(psi_sorted.index, psi_sorted.values, color=colors)
plt.axvline(0.1, color="#854F0B", linestyle="--", linewidth=1)
plt.axvline(0.2, color="#A32D2D", linestyle="--", linewidth=1)
plt.title("Population Stability Index by feature\n(simulated forward period vs. validation sample)")
plt.tight_layout()
plt.savefig(CHART + "psi.png", dpi=150)
plt.close()

# ---------- 5. Fairness / bias audit ----------
fairness = {}
for attr in ["protected_group", "gender"]:
    sf = test[attr]
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_test, y_pred=pred, sensitive_features=sf,
    )
    dp_diff = demographic_parity_difference(y_test, pred, sensitive_features=sf)
    dp_ratio = demographic_parity_ratio(y_test, pred, sensitive_features=sf)
    eo_diff = equalized_odds_difference(y_test, pred, sensitive_features=sf)

    fairness[attr] = {
        "by_group": mf.by_group.round(4).to_dict(),
        "demographic_parity_difference": round(float(dp_diff), 4),
        "demographic_parity_ratio": round(float(dp_ratio), 4),
        "equalized_odds_difference": round(float(eo_diff), 4),
        "four_fifths_pass": bool(dp_ratio >= 0.8),
    }

    grp = mf.by_group["selection_rate"]
    plt.figure(figsize=(5, 3.5))
    bars = plt.bar(grp.index.astype(str), grp.values, color=["#185FA5", "#D85A30"])
    plt.axhline(grp.values.max() * 0.8, color="#854F0B", linestyle="--", linewidth=1,
                label="80% (four-fifths) threshold")
    plt.ylabel("Model approval (selection) rate")
    plt.title(f"Approval rate by {attr.replace('_', ' ')}")
    plt.legend(fontsize=8)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART + f"fairness_{attr}.png", dpi=150)
    plt.close()

results["fairness"] = fairness

# ---------- 6. Data quality ----------
raw = pd.read_csv("/home/claude/ai_model_validation_project/data/loan_applications.csv")
dq = {
    "n_rows": int(len(raw)),
    "n_features": len(features),
    "missing_by_feature": raw[features].isna().mean().round(4).to_dict(),
    "duplicate_applicant_ids": int(raw["applicant_id"].duplicated().sum()),
    "class_balance": raw["loan_approved"].value_counts(normalize=True).round(4).to_dict(),
    "credit_score_range": [int(raw["credit_score"].min()), int(raw["credit_score"].max())],
    "outliers_income_gt_3std": int((np.abs(raw["annual_income"] - raw["annual_income"].mean()) >
                                     3 * raw["annual_income"].std()).sum()),
}
results["data_quality"] = dq

with open(OUT + "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results["performance"], indent=2))
print(json.dumps(results["fairness"]["protected_group"], indent=2))
print(json.dumps(results["fairness"]["gender"], indent=2))
print(json.dumps(results["stability"], indent=2))
