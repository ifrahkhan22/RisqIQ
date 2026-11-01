import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

TARGETS = ["target_turbulence", "target_next5"]

df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["date"])
features = [c for c in df.columns if c not in ["date"] + TARGETS]

train = df[df["date"] < "2023-01-01"]
valid = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"]

def report(name, y, score, predicted):
    print(f"  {name:22s} PR-AUC={average_precision_score(y, score):.3f}  "
          f"precision={precision_score(y, predicted, zero_division=0):.3f}  "
          f"recall={recall_score(y, predicted):.3f}")

for target in TARGETS:
    print("\n" + "=" * 60)
    print("TARGET:", target)
    X_train, y_train = train[features], train[target]

    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    logit.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                class_weight="balanced_subsample", random_state=42)
    rf.fit(X_train, y_train)

    for split_name, part in [("VALIDATION 2023-24", valid), ("TEST 2025+", test)]:
        X, y = part[features], part[target]
        print(f"\n{split_name}: {len(part)} rows | event rate {y.mean():.3f} "
              "(= PR-AUC of random guessing)")
        # Baseline: "volatility is above its normal level, so expect trouble"
        base_pred = X["vol_20d"] >= X_train["vol_20d"].median()
        report("Baseline (vol high)", y, X["vol_20d"], base_pred)
        report("Logistic regression", y, logit.predict_proba(X)[:, 1], logit.predict(X))
        report("Random forest", y, rf.predict_proba(X)[:, 1], rf.predict(X))