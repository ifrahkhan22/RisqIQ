import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

df = pd.read_csv("data/processed/model_dataset_ext.csv", parse_dates=["date"])

TARGETS = ["target_turbulence", "target_next5"]
gpr_cols = ["gpr", "gpr_ma7", "gpr_ma30", "gpr_ma7_vs_ma30", "gpr_threat_ma7"]
ovx_cols = ["ovx", "ovx_chg_5d"]
price = [c for c in df.columns if c not in ["date"] + TARGETS + gpr_cols + ovx_cols]

experiments = [
    ("FULL 2015+", df,
     {"price": price, "price + GPR": price + gpr_cols}),
    ("OVX WINDOW 2021-10+", df[df["ovx"].notna() & df["ovx_chg_5d"].notna()],
     {"price": price, "price + OVX": price + ovx_cols,
      "price + GPR + OVX": price + gpr_cols + ovx_cols}),
]

rows = []
for window, data, sets in experiments:
    train = data[data["date"] < "2023-01-01"]
    parts = {
        "valid": data[(data["date"] >= "2023-01-01") & (data["date"] < "2025-01-01")],
        "test": data[data["date"] >= "2025-01-01"],
    }
    for target in TARGETS:
        # Simple rule: higher recent volatility = higher risk
        row = {"window": window, "target": target, "features": "(rule: vol_20d)",
               "model": "rule", "train_rows": len(train)}
        for p, part in parts.items():
            row[p + "_prauc"] = round(average_precision_score(part[target], part["vol_20d"]), 3)
            row[p + "_random"] = round(part[target].mean(), 3)
        rows.append(row)

        for set_name, cols in sets.items():
            models = {
                "logistic": make_pipeline(
                    StandardScaler(),
                    LogisticRegression(class_weight="balanced", max_iter=1000)),
                "forest": RandomForestClassifier(
                    n_estimators=300, max_depth=5, min_samples_leaf=20,
                    class_weight="balanced_subsample", random_state=42),
            }
            for model_name, m in models.items():
                m.fit(train[cols], train[target])
                row = {"window": window, "target": target, "features": set_name,
                       "model": model_name, "train_rows": len(train)}
                for p, part in parts.items():
                    score = m.predict_proba(part[cols])[:, 1]
                    row[p + "_prauc"] = round(average_precision_score(part[target], score), 3)
                    row[p + "_random"] = round(part[target].mean(), 3)
                rows.append(row)

pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False))