import matplotlib
matplotlib.use("Agg")                      # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

Path("outputs").mkdir(exist_ok=True)

ext = pd.read_csv("data/processed/model_dataset_ext.csv", parse_dates=["date"])
mkt = pd.read_csv("data/processed/brent_market_features.csv", parse_dates=["date"])

TARGET = "target_next5"                    # "surprise shock in the next 5 days"
drop = ["date", "target_next5", "target_turbulence", "ovx", "ovx_chg_5d"]
features = [c for c in ext.columns if c not in drop]   # price features + GPR

# Train on 2015-2024, test on 2025 onward
train = ext[ext["date"] < "2025-01-01"]
test = ext[ext["date"] >= "2025-01-01"].copy()

rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                            class_weight="balanced_subsample", random_state=42)
rf.fit(train[features], train[TARGET])
test["risk_prob"] = rf.predict_proba(test[features])[:, 1]
test = test.merge(
    mkt[["date", "price_usd_per_barrel", "vol_shock", "annualized_volatility_20d_pct"]],
    on="date", how="left")

# flag the 20% highest-risk days
cut = test["risk_prob"].quantile(0.80)
flagged = test[test["risk_prob"] >= cut]
print("Test days:", len(test))
print("PR-AUC:", round(average_precision_score(test[TARGET], test["risk_prob"]), 3),
      "| random guess:", round(test[TARGET].mean(), 3))
print("Flagging the top 20% riskiest days:")
print("  share of flagged days followed by a shock:", round(flagged[TARGET].mean(), 3),
      "| share on all days:", round(test[TARGET].mean(), 3))
print("  share of all shocks that were caught:",
      round(flagged[TARGET].sum() / test[TARGET].sum(), 3))
test[["date", "risk_prob", TARGET]].to_csv("outputs/test_predictions.csv", index=False)

imp = permutation_importance(rf, test[features], test[TARGET],
                             scoring="average_precision", n_repeats=10, random_state=42)
ranking = pd.Series(imp.importances_mean, index=features).sort_values()
print("\nFeature importance:\n", ranking.sort_values(ascending=False).round(4).to_string())

# Value-at-Risk number for fuel buyers (historical, not a forecast)
ret5 = mkt["price_usd_per_barrel"].pct_change(5) * 100
last_year = ret5.tail(250)
print("\nOver the last 250 trading days, 1 five-day period in 20 saw a price RISE of at least",
      round(last_year.quantile(0.95), 1), "%")
print("Over all history since 2015, that figure is",
      round(ret5.quantile(0.95), 1), "%")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(mkt["date"], mkt["price_usd_per_barrel"], color="#1f4e79", lw=1)
s = mkt[mkt["vol_shock"] == True]
ax.scatter(s["date"], s["price_usd_per_barrel"], color="red", s=10, label="Volatility-adjusted shock")
ax.set_title("Brent spot price and detected shock days"); ax.set_ylabel("USD per barrel")
ax.legend(); fig.tight_layout(); fig.savefig("outputs/chart1_price_and_shocks.png", dpi=150); plt.close(fig)

o = ext[["date", "ovx"]].dropna().merge(
    mkt[["date", "annualized_volatility_20d_pct"]], on="date")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(o["date"], o["annualized_volatility_20d_pct"], label="Realized 20-day volatility (ours)")
ax.plot(o["date"], o["ovx"], label="OVX (market's fear index)", alpha=0.8)
ax.set_title("Our volatility measure vs OVX"); ax.set_ylabel("Percent")
ax.legend(); fig.tight_layout(); fig.savefig("outputs/chart2_volatility_vs_ovx.png", dpi=150); plt.close(fig)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
a1.plot(test["date"], test["price_usd_per_barrel"], color="#1f4e79", lw=1)
sh = test[test["vol_shock"] == True]
a1.scatter(sh["date"], sh["price_usd_per_barrel"], color="red", s=14, label="Actual shock day")
a1.set_ylabel("USD per barrel"); a1.legend(); a1.set_title("Test period: price and model risk score")
a2.plot(test["date"], test["risk_prob"], color="darkorange", lw=1)
a2.axhline(cut, ls="--", color="grey", label="Top-20% alert line")
a2.set_ylabel("Risk score (next 5 days)"); a2.legend()
fig.tight_layout(); fig.savefig("outputs/chart3_test_risk_score.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ranking.plot(kind="barh", ax=ax, color="#1f4e79")
ax.set_title("Which inputs mattered most"); fig.tight_layout()
fig.savefig("outputs/chart4_feature_importance.png", dpi=150); plt.close(fig)
print("\nCharts saved in the outputs folder.")