import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# LOAD + CLEAN DATA
# =============================

# Skip the two descriptive header rows
df = pd.read_csv(
    "trumpevents.csv",
    skiprows=2,
    encoding="latin1"
)

# Rename first column explicitly
df = df.rename(columns={df.columns[0]: "Day"})

# Parse dates safely
df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
df = df.dropna(subset=["Day"])
df = df.set_index("Day")

# Convert all values to numeric
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# =============================
# METRIC FUNCTIONS
# =============================

def peak_magnitude(series):
    return series.max()

def attention_half_life(series):
    peak = series.max()
    if peak == 0:
        return np.nan

    peak_day = series.idxmax()
    half_level = 0.5 * peak
    post_peak = series.loc[peak_day:]
    below_half = post_peak[post_peak <= half_level]

    if below_half.empty:
        return np.nan
    return (below_half.index[0] - peak_day).days

def total_attention_auc(series):
    return np.trapz(series.values)

def decay_lambda(series):
    hl = attention_half_life(series)
    if hl is None or np.isnan(hl) or hl <= 0:
        return np.nan
    return np.log(2) / hl

# =============================
# COMPUTE METRICS
# =============================

results = []

for col in df.columns:
    s = df[col]
    results.append({
        "event": col,
        "peak": peak_magnitude(s),
        "half_life_days": attention_half_life(s),
        "total_attention": total_attention_auc(s),
        "decay_lambda": decay_lambda(s)
    })

metrics = pd.DataFrame(results)

# Drop dead events
metrics = metrics[metrics["total_attention"] > 0]

# =============================
# REGIME CLASSIFICATION
# =============================

metrics["regime"] = np.where(
    metrics["half_life_days"] >= 2,
    "persistent",
    "shock"
)

print("\n=== REGIME COUNTS ===")
print(metrics["regime"].value_counts())

# =============================
# REGRESSION FUNCTION
# =============================

def run_regression(df, label):
    X = df["peak"].values
    y = df["total_attention"].values

    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

    y_hat = X_design @ beta
    r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)

    print(f"\n=== {label.upper()} REGIME ===")
    print("AUC ~ Peak")
    print(f"Intercept: {beta[0]:.3f}")
    print(f"Slope: {beta[1]:.3f}")
    print(f"R² = {r2:.3f}")

# =============================
# RUN REGRESSIONS
# =============================

run_regression(metrics[metrics["regime"] == "shock"], "shock")
run_regression(metrics[metrics["regime"] == "persistent"], "persistent")

# =============================
# PLOTS
# =============================

# Peak vs AUC by regime
plt.figure()
for r, c in zip(["shock", "persistent"], ["blue", "red"]):
    sub = metrics[metrics["regime"] == r]
    plt.scatter(sub["peak"], sub["total_attention"], label=r, color=c)

plt.xlabel("Peak Attention")
plt.ylabel("Total Attention (AUC)")
plt.title("Attention Regimes: Shock vs Persistent")
plt.legend()
plt.show()

# =============================
# OUTPUT TABLE
# =============================

metrics = metrics.sort_values("total_attention", ascending=False)

print("\n=== ATTENTION LIFESPAN METRICS ===\n")
print(metrics[[
    "event",
    "peak",
    "half_life_days",
    "total_attention",
    "decay_lambda",
    "regime"
]])