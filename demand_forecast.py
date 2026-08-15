"""
COSC2669/COSC2816 Individual Task 1 - Part 1.3 Data Analysis
Half-hourly electricity demand forecasting for a NEM retailer (Alinta Energy case).

Datasets
  A) AEMO Aggregated Price and Demand Data, VIC1 region (half-hourly TOTALDEMAND, RRP)
     https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
     Download the monthly CSVs (PRICE_AND_DEMAND_YYYYMM_VIC1.csv) into ./data/aemo/
  B) UCI Individual Household Electric Power Consumption (1-min, Sceaux France)
     https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
     Put household_power_consumption.txt into ./data/uci/

Models
  M1) Histogram Gradient Boosting Regressor (tabular, lag/calendar features)
  M2) Multilayer Perceptron (same feature matrix, standardised)
  Baseline) Seasonal naive (demand at t-48, i.e. same time yesterday)

Run:  python demand_forecast.py
Outputs: results/metrics.csv, results/metrics_table.tex, results/*.png, console summary
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RNG = 42
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_aemo(folder="data/aemo"):
    """Concatenate AEMO monthly VIC1 price/demand CSVs into one half-hourly series."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No AEMO CSVs found in {folder}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = (df[["SETTLEMENTDATE", "TOTALDEMAND", "RRP"]]
          .drop_duplicates("SETTLEMENTDATE")
          .sort_values("SETTLEMENTDATE")
          .set_index("SETTLEMENTDATE"))
    # AEMO publishes at 5-min in some files; force a clean 30-min grid
    df = df.resample("30min").mean()
    df["TOTALDEMAND"] = df["TOTALDEMAND"].interpolate(limit=4)
    print(f"[AEMO]  {len(df):,} half-hourly rows  "
          f"{df.index.min().date()} to {df.index.max().date()}  "
          f"missing demand: {df['TOTALDEMAND'].isna().sum()}")
    return df


def load_uci(path="data/uci/household_power_consumption.txt"):
    """Load the UCI household series and aggregate 1-min -> 30-min mean kW."""
    # NOTE: pandas removed the dict form of parse_dates in 2.2+, so the
    # timestamp is built explicitly from the separate Date and Time columns.
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False,
                     dtype={"Date": "string", "Time": "string"})
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                              format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt"]).drop(columns=["Date", "Time"])
    df = df.set_index("dt").sort_index()
    num = ["Global_active_power", "Global_reactive_power", "Voltage",
           "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    hh = df[num].resample("30min").mean()
    hh = hh.rename(columns={"Global_active_power": "hh_kw"})
    print(f"[UCI]   {len(hh):,} half-hourly rows  "
          f"{hh.index.min().date()} to {hh.index.max().date()}  "
          f"missing: {hh['hh_kw'].isna().mean():.2%}")
    return hh


# ----------------------------------------------------------------------------
# 2. Feature engineering (shared by both models)
# ----------------------------------------------------------------------------
def make_features(s, name="y"):
    """Lag, rolling and calendar features for a half-hourly series."""
    X = pd.DataFrame(index=s.index)
    X[name] = s

    # Autoregressive lags: 30min, 1h, same-time-yesterday, same-time-last-week
    for lag in [1, 2, 3, 48, 96, 336]:
        X[f"lag_{lag}"] = s.shift(lag)

    # Rolling statistics over the previous day / week (shifted to avoid leakage)
    X["roll_mean_48"] = s.shift(1).rolling(48).mean()
    X["roll_std_48"] = s.shift(1).rolling(48).std()
    X["roll_max_48"] = s.shift(1).rolling(48).max()
    X["roll_mean_336"] = s.shift(1).rolling(336).mean()

    # Calendar
    idx = X.index
    X["tod"] = idx.hour * 2 + idx.minute // 30      # 0..47 settlement period
    X["dow"] = idx.dayofweek
    X["month"] = idx.month
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Fourier terms let the MLP express daily/weekly/annual cycles smoothly
    X["sin_day"] = np.sin(2 * np.pi * X["tod"] / 48)
    X["cos_day"] = np.cos(2 * np.pi * X["tod"] / 48)
    X["sin_week"] = np.sin(2 * np.pi * (X["dow"] * 48 + X["tod"]) / 336)
    X["cos_week"] = np.cos(2 * np.pi * (X["dow"] * 48 + X["tod"]) / 336)
    doy = idx.dayofyear
    X["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * doy / 365.25)

    return X.dropna()


def chrono_split(X, target, test_frac=0.2):
    """Chronological hold-out - never shuffle a time series."""
    n_test = int(len(X) * test_frac)
    train, test = X.iloc[:-n_test], X.iloc[-n_test:]
    feats = [c for c in X.columns if c != target]
    return (train[feats], train[target], test[feats], test[target])


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, label, extra=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    row = {"model": label, "MAE": mae, "RMSE": rmse, "MAPE_pct": mape}
    if extra:
        row.update(extra)
    return row


def peak_mae(y_true, y_pred, index):
    """MAE restricted to the evening peak (16:00-20:00), when spot risk concentrates."""
    mask = (index.hour >= 16) & (index.hour < 20)
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])


def cost_proxy(y_true, y_pred, rrp):
    """Rough $/interval imbalance cost: |error| MW * 0.5 h * spot price $/MWh."""
    return float(np.mean(np.abs(y_true - y_pred) * 0.5 * rrp))


# ----------------------------------------------------------------------------
# 4. Model runner
# ----------------------------------------------------------------------------
def run_models(X, target, tag, rrp=None):
    Xtr, ytr, Xte, yte = chrono_split(X, target)
    print(f"\n=== {tag}: train {len(Xtr):,} / test {len(Xte):,} intervals ===")
    rows, preds = [], {}

    # Baseline: seasonal naive (same time yesterday)
    naive = Xte["lag_48"].values
    rows.append(evaluate(yte.values, naive, f"{tag} | Seasonal naive (t-48)"))
    preds["Seasonal naive"] = naive

    # M1 Gradient boosting
    gbm = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_depth=None,
        max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=RNG)
    gbm.fit(Xtr, ytr)
    p_gbm = gbm.predict(Xte)
    rows.append(evaluate(yte.values, p_gbm, f"{tag} | Gradient boosting"))
    preds["Gradient boosting"] = p_gbm

    # M2 MLP (scaled inputs; the tree model does not need scaling, the net does)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                     alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
                     max_iter=300, early_stopping=True, n_iter_no_change=15,
                     random_state=RNG))
    mlp.fit(Xtr, ytr)
    p_mlp = mlp.predict(Xte)
    rows.append(evaluate(yte.values, p_mlp, f"{tag} | MLP"))
    preds["MLP"] = p_mlp

    # Operational metrics only meaningful for the grid-level series
    if rrp is not None:
        rrp_te = rrp.reindex(Xte.index).ffill().values
        for name, p in preds.items():
            for r in rows:
                if r["model"].endswith(name):
                    r["peak_MAE"] = peak_mae(yte.values, p, Xte.index)
                    r["imbalance_$_per_interval"] = cost_proxy(yte.values, p, rrp_te)

    for r in rows:
        print("  " + " | ".join(
            f"{k}={v:,.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.items()))

    # Diagnostic plot: one week of the test period
    fig, ax = plt.subplots(figsize=(11, 4))
    win = slice(0, 336)
    ax.plot(Xte.index[win], yte.values[win], label="Actual", lw=1.6, color="black")
    for name, p in preds.items():
        ax.plot(Xte.index[win], p[win], label=name, lw=1.1, alpha=0.85)
    ax.set_title(f"{tag}: one week of the hold-out period")
    ax.set_ylabel(target)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/{tag.lower().replace(' ', '_')}_week.png", dpi=150)
    plt.close(fig)

    # Error by time of day - shows *where* each model fails
    err = pd.DataFrame({"tod": Xte["tod"].values})
    for name, p in preds.items():
        err[name] = np.abs(yte.values - p)
    prof = err.groupby("tod").mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    prof.plot(ax=ax)
    ax.set_xlabel("Settlement period (0 = 00:00)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"{tag}: error profile across the day")
    fig.tight_layout()
    fig.savefig(f"results/{tag.lower().replace(' ', '_')}_error_profile.png", dpi=150)
    plt.close(fig)

    return rows


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    all_rows = []

    aemo = load_aemo()
    Xg = make_features(aemo["TOTALDEMAND"].dropna(), name="demand_mw")
    all_rows += run_models(Xg, "demand_mw", "Grid VIC", rrp=aemo["RRP"])

    try:
        uci = load_uci()
        Xh = make_features(uci["hh_kw"].dropna(), name="hh_kw")
        all_rows += run_models(Xh, "hh_kw", "Household")
    except FileNotFoundError as e:
        print(f"[UCI] skipped: {e}")

    res = pd.DataFrame(all_rows)
    res.to_csv("results/metrics.csv", index=False)
    with open("results/metrics_table.tex", "w") as f:
        f.write(res.round(3).to_latex(index=False, escape=True))
    print("\nSaved results/metrics.csv, results/metrics_table.tex and plots in results/")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
