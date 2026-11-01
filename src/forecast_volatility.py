import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

Path("outputs").mkdir(exist_ok=True)
H = 5                          # forecast horizon: the next 5 trading days
EMBARGO = pd.Timedelta(days=10)  # gap between training and test (targets look 5 days ahead)
MIN_TRAIN = 200
FLOOR = 0.05                   # avoids log(0) on days with zero movement

# ------------------------------------------------------------------ data
mkt = pd.read_csv("data/processed/brent_market_features.csv", parse_dates=["date"])
mkt = mkt.sort_values("date").reset_index(drop=True)
r = mkt["daily_return_pct"]

def log_floor(x):
    return np.log(np.maximum(x, FLOOR))

# What we know today: yesterday's move, last week's volatility, last month's volatility
mkt["log_rv_d"] = log_floor(r.abs())
mkt["log_rv_w"] = log_floor(np.sqrt((r ** 2).rolling(5).mean()))
mkt["log_rv_m"] = log_floor(np.sqrt((r ** 2).rolling(22).mean()))

# What we want to predict: volatility over the NEXT 5 trading days
mkt["y"] = log_floor(np.sqrt((r ** 2).rolling(H).mean().shift(-H)))
mkt["fwd5_ret"] = (mkt["price_usd_per_barrel"].shift(-H) / mkt["price_usd_per_barrel"] - 1) * 100

# "Static" risk band: 95th/5th percentile of the past 250 five-day moves
ret5_back = mkt["price_usd_per_barrel"].pct_change(H) * 100
mkt["static_up"] = ret5_back.rolling(250).quantile(0.95)
mkt["static_low"] = ret5_back.rolling(250).quantile(0.05)

# News signal (GPR, daily)
gpr = pd.read_excel("data/raw/data_gpr_daily_recent.xls")
gpr["date"] = pd.to_datetime(gpr["DAY"].astype(int).astype(str), format="%Y%m%d")
gpr["log_gpr_ma7"] = np.log(gpr["GPRD_MA7"])
gpr["log_gpr_ma30"] = np.log(gpr["GPRD_MA30"])
gpr["gpr_trend"] = gpr["log_gpr_ma7"] - gpr["log_gpr_ma30"]
mkt = mkt.merge(gpr[["date", "log_gpr_ma7", "log_gpr_ma30", "gpr_trend"]], on="date", how="left")

# Market fear signal (OVX, only from 2021-10-22)
ovx = pd.read_csv("data/raw/OVXCLS.csv", parse_dates=["observation_date"])
ovx = ovx.rename(columns={"observation_date": "date", "OVXCLS": "ovx"})
ovx["ovx"] = pd.to_numeric(ovx["ovx"], errors="coerce")
ovx = ovx.dropna().sort_values("date")
ovx["log_ovx"] = np.log(ovx["ovx"])
mkt = pd.merge_asof(mkt.sort_values("date"), ovx[["date", "log_ovx"]], on="date",
                    direction="backward", tolerance=pd.Timedelta(days=5))

HAR = ["log_rv_d", "log_rv_w", "log_rv_m"]
GPR = ["log_gpr_ma7", "log_gpr_ma30", "gpr_trend"]
OVX = ["log_ovx"]

# part A
# Does a model beat "next week's volatility = last month's volatility"?
def run_experiment(title, data, models, years):
    sse = {}     # model -> [sse_model, sse_bench]
    per_year = {}
    for Y in years:
        start = pd.Timestamp(f"{Y}-01-01")
        end = pd.Timestamp(f"{Y + 1}-01-01")
        train = data[data["date"] < start - EMBARGO]
        test = data[(data["date"] >= start) & (data["date"] < end)]
        if len(train) < MIN_TRAIN or len(test) < 20:
            continue
        bench = ((test["y"] - test["log_rv_m"]) ** 2).sum()
        for name, cols in models.items():
            model = LinearRegression().fit(train[cols], train["y"])
            err = ((test["y"] - model.predict(test[cols])) ** 2).sum()
            per_year.setdefault(name, {})[Y] = round(1 - err / bench, 2)
            tot = sse.setdefault(name, [0.0, 0.0])
            tot[0] += err
            tot[1] += bench
    print("\n" + "=" * 70)
    print(title)
    print("Skill = how much smaller the error is than the 'last month' benchmark")
    print("(0 = same, +0.10 = 10% smaller error, negative = worse)\n")
    table = pd.DataFrame(per_year).T
    table["pooled"] = [round(1 - sse[n][0] / sse[n][1], 3) for n in table.index]
    table["years_better"] = [f"{int((table.loc[n].drop(['pooled']) > 0).sum())}/"
                             f"{int(table.loc[n].drop(['pooled']).notna().sum())}"
                             for n in table.index]
    print(table.to_string())

full = mkt.dropna(subset=HAR + GPR + ["y"])
run_experiment("FULL HISTORY (test years 2017-2026, retrained every year)", full,
               {"HAR": HAR, "HAR + GPR": HAR + GPR}, range(2017, 2027))

win = mkt.dropna(subset=HAR + GPR + OVX + ["y"])
run_experiment("OVX WINDOW (test years 2023-2026; small training sets)", win,
               {"HAR": HAR, "HAR + OVX": HAR + OVX,
                "HAR + GPR": HAR + GPR, "HAR + GPR + OVX": HAR + GPR + OVX},
               range(2023, 2027))

# part B
# Turn the volatility forecast into a 95% risk band and check how often reality breaks it
BAND = HAR     # keep simple; switch to HAR + GPR only if part A shows a consistent gain
band_data = mkt.dropna(subset=BAND + ["y", "fwd5_ret", "static_up", "static_low"])
rows, preds = [], []
for Y in range(2017, 2027):
    start = pd.Timestamp(f"{Y}-01-01")
    end = pd.Timestamp(f"{Y + 1}-01-01")
    train = band_data[band_data["date"] < start - EMBARGO]
    test = band_data[(band_data["date"] >= start) & (band_data["date"] < end)]
    if len(train) < MIN_TRAIN or len(test) < 20:
        continue
    model = LinearRegression().fit(train[BAND], train["y"])
    sig_train = np.exp(model.predict(train[BAND]))
    z = train["fwd5_ret"] / (sig_train * np.sqrt(H))      # how big were moves vs forecast?
    q_hi, q_lo = z.quantile(0.95), z.quantile(0.05)
    sig = np.exp(model.predict(test[BAND]))
    up, low = q_hi * sig * np.sqrt(H), q_lo * sig * np.sqrt(H)
    rows.append({
        "year": Y, "days": len(test),
        "dynamic_up_%": round((test["fwd5_ret"].values > up).mean() * 100, 1),
        "static_up_%": round((test["fwd5_ret"] > test["static_up"]).mean() * 100, 1),
        "dynamic_low_%": round((test["fwd5_ret"].values < low).mean() * 100, 1),
        "static_low_%": round((test["fwd5_ret"] < test["static_low"]).mean() * 100, 1),
    })
    preds.append(pd.DataFrame({"date": test["date"].values, "forecast_vol": sig,
                               "actual_vol": np.exp(test["y"].values)}))

print("\n" + "=" * 70)
print("RISK BAND BACKTEST: how often did the next 5 days break the 95% band?")
print("Target is about 5.0% for each column. Far above 5% = band too narrow.\n")
bt = pd.DataFrame(rows)
print(bt.to_string(index=False))
w = bt["days"]
print("\nAll years (weighted):",
      {c: round(float((bt[c] * w).sum() / w.sum()), 1)
       for c in ["dynamic_up_%", "static_up_%", "dynamic_low_%", "static_low_%"]})

# Latest risk band, using ALL available history
last = mkt.dropna(subset=BAND).tail(1)
train = band_data[band_data["date"] < last["date"].iloc[0] - EMBARGO]
model = LinearRegression().fit(train[BAND], train["y"])
z = train["fwd5_ret"] / (np.exp(model.predict(train[BAND])) * np.sqrt(H))
sig = float(np.exp(model.predict(last[BAND]))[0])
print("\nLATEST (", last["date"].iloc[0].date(), ")")
print("  expected daily move size:", round(sig, 2), "% | annualized:", round(sig * 252 ** 0.5, 1), "%")
print("  95% band for the next 5 trading days: from",
      round(z.quantile(0.05) * sig * np.sqrt(H), 1), "% to +",
      round(z.quantile(0.95) * sig * np.sqrt(H), 1), "%")

# Chart
p = pd.concat(preds)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_yscale("log")
ax.plot(p["date"], p["actual_vol"] * 252 ** 0.5, label="Actual next-5-day volatility", alpha=0.6)
ax.plot(p["date"], p["forecast_vol"] * 252 ** 0.5, label="Forecast (made before)", lw=1.2)
ax.set_title("Volatility forecast vs outcome (walk-forward, annualized %)")
ax.set_ylabel("Percent"); ax.legend(); fig.tight_layout()
fig.savefig("outputs/chart5_volatility_forecast.png", dpi=150); plt.close(fig)
p.to_csv("outputs/volatility_forecasts.csv", index=False)
print("\nSaved outputs/chart5_volatility_forecast.png and outputs/volatility_forecasts.csv")

# part C
# Same risk-band test, now with the OVX signal (it helped in all 4 test years in part A)
def band_backtest(data, cols, years):
    out = []
    for Y in years:
        start = pd.Timestamp(f"{Y}-01-01")
        end = pd.Timestamp(f"{Y + 1}-01-01")
        train = data[data["date"] < start - EMBARGO]
        test = data[(data["date"] >= start) & (data["date"] < end)]
        if len(train) < MIN_TRAIN or len(test) < 20:
            continue
        model = LinearRegression().fit(train[cols], train["y"])
        z = train["fwd5_ret"] / (np.exp(model.predict(train[cols])) * np.sqrt(H))
        sig = np.exp(model.predict(test[cols]))
        up = z.quantile(0.95) * sig * np.sqrt(H)
        low = z.quantile(0.05) * sig * np.sqrt(H)
        out.append({"year": Y, "days": len(test),
                    "up_%": (test["fwd5_ret"].values > up).mean() * 100,
                    "low_%": (test["fwd5_ret"].values < low).mean() * 100,
                    "static_up_%": (test["fwd5_ret"] > test["static_up"]).mean() * 100,
                    "static_low_%": (test["fwd5_ret"] < test["static_low"]).mean() * 100})
    return pd.DataFrame(out)

ovx_band_data = mkt.dropna(subset=HAR + OVX + ["y", "fwd5_ret", "static_up", "static_low"])
for label, cols in [("HAR only (same short window)", HAR), ("HAR + OVX", HAR + OVX)]:
    t = band_backtest(ovx_band_data, cols, range(2023, 2027))
    print("\n" + "=" * 70)
    print("RISK BAND BACKTEST 2023-2026:", label, "(target about 5.0% in each column)")
    print(t.round(1).to_string(index=False))
    w = t["days"]
    print("weighted:", {c: round(float((t[c] * w).sum() / w.sum()), 1)
                        for c in ["up_%", "low_%", "static_up_%", "static_low_%"]})

# Latest band using HAR + OVX
cols = HAR + OVX
last = mkt.dropna(subset=cols).tail(1)
train = ovx_band_data[ovx_band_data["date"] < last["date"].iloc[0] - EMBARGO]
model = LinearRegression().fit(train[cols], train["y"])
z = train["fwd5_ret"] / (np.exp(model.predict(train[cols])) * np.sqrt(H))
sig = float(np.exp(model.predict(last[cols]))[0])
print("\nLATEST with OVX (", last["date"].iloc[0].date(), ")")
print("  expected daily move size:", round(sig, 2), "% | annualized:", round(sig * 252 ** 0.5, 1), "%")
print("  95% band for the next 5 trading days: from",
      round(z.quantile(0.05) * sig * np.sqrt(H), 1), "% to +",
      round(z.quantile(0.95) * sig * np.sqrt(H), 1), "%")
print("  static band (past 250 days):", round(float(last["static_low"].iloc[0]), 1), "% to +",
      round(float(last["static_up"].iloc[0]), 1), "%")