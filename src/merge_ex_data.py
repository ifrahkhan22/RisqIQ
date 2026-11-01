import pandas as pd

model = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["date"])
model = model.sort_values("date")

# GPR (daily, every calendar day)
gpr = pd.read_excel("data/raw/data_gpr_daily_recent.xls")
gpr["date"] = pd.to_datetime(gpr["DAY"].astype(int).astype(str), format="%Y%m%d")
gpr = gpr.sort_values("date")
gpr["gpr_threat_ma7"] = gpr["GPRD_THREAT"].rolling(7, min_periods=1).mean()
gpr = gpr.rename(columns={"GPRD": "gpr", "GPRD_MA7": "gpr_ma7", "GPRD_MA30": "gpr_ma30"})
gpr["gpr_ma7_vs_ma30"] = gpr["gpr_ma7"] / gpr["gpr_ma30"]   # is risk rising?
gpr = gpr[["date", "gpr", "gpr_ma7", "gpr_ma30", "gpr_ma7_vs_ma30", "gpr_threat_ma7"]]
model = model.merge(gpr, on="date", how="left")

# OVX (daily, only from 2021-10-22)
ovx = pd.read_csv("data/raw/OVXCLS.csv", parse_dates=["observation_date"])
ovx = ovx.rename(columns={"observation_date": "date", "OVXCLS": "ovx"})
ovx["ovx"] = pd.to_numeric(ovx["ovx"], errors="coerce")   # FRED marks gaps with "."
ovx = ovx.dropna().sort_values("date")
ovx["ovx_chg_5d"] = ovx["ovx"].pct_change(5) * 100
model = pd.merge_asof(model, ovx, on="date", direction="backward",
                      tolerance=pd.Timedelta(days=5))

model.to_csv("data/processed/model_dataset_ext.csv", index=False)

print("Rows:", len(model))
print("Rows with GPR:", model["gpr"].notna().sum())
print("Rows with OVX:", model["ovx"].notna().sum(),
      "| first OVX date:", model.loc[model["ovx"].notna(), "date"].min().date())
print(model[["date", "gpr", "gpr_ma7", "ovx", "ovx_chg_5d"]].tail(5).to_string(index=False))