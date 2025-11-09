# filename: nfip_wind_lr_hurricane_45degfilter.py
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "nfip_with_eventtype.csv"
EVENT_TYPE_COL = "event_type"
EVENT_KEEP = "Hurricane"

X_COL = "wind_max_m_s"
TARGETS = [
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "totalPaid",
]

# Distance tolerance to the 45° line *after robust z-scaling*.
# This is *perpendicular distance* to y=x in the z-space.
PERP_EPS = 0.5   # try 0.3 (stricter) or 0.7 (looser)

PLOT_DIR = "plots_wind_lr_hurricane_45deg"
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def clean_numeric(df, cols):
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=cols)

def robust_z(v):
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    sigma = 1.4826 * mad if mad > 0 else (np.std(v) + 1e-9)
    return (v - med) / (sigma + 1e-9)

def plot_scatter_with_lr(x, y, title, xlabel, ylabel, outpath, model=None):
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, alpha=0.6)
    if model is not None and len(x) > 1:
        xx = np.linspace(np.min(x), np.max(x), 200).reshape(-1, 1)
        yy = model.predict(xx)
        plt.plot(xx, yy)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_zspace_with_y_eq_x(zx, zy, title, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(zx, zy, alpha=0.5)
    lo = min(np.min(zx), np.min(zy))
    hi = max(np.max(zx), np.max(zy))
    grid = np.linspace(lo, hi, 200)
    plt.plot(grid, grid)  # y = x (45° line)
    plt.title(title)
    plt.xlabel(f"Z({X_COL})")
    plt.ylabel("Z(target)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ----------------------------
# Load + filter to Hurricane
# ----------------------------
df = pd.read_csv(CSV_PATH)
if EVENT_TYPE_COL not in df.columns:
    raise ValueError(f"Column '{EVENT_TYPE_COL}' not found.")

df[EVENT_TYPE_COL] = df[EVENT_TYPE_COL].astype(str).str.strip()
df = df[df[EVENT_TYPE_COL].str.casefold() == EVENT_KEEP.casefold()].copy()
print(f"Rows after event filter ({EVENT_TYPE_COL} == '{EVENT_KEEP}'): {len(df):,}")

needed = [X_COL] + TARGETS
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = clean_numeric(df, needed)
print(f"Rows after numeric cleaning: {len(df):,}")
if len(df) < 10:
    raise SystemExit("Too few rows to proceed.")

# ----------------------------
# For each target:
# - robust z-scale X and Y
# - keep points with perpendicular distance to y=x <= PERP_EPS
#   (in z-space, perpendicular distance = |zy - zx| / sqrt(2))
# - plot z-space (with y=x) and original-space with LR
# - 80/20 LR on kept points (original units)
# ----------------------------
metrics = []
rt2 = math.sqrt(2.0)

for target in TARGETS:
    sub = df[[X_COL, target]].dropna().copy()
    if len(sub) < 10:
        print(f"[WARN] Not enough rows for {target}; skipping.")
        continue

    x = sub[X_COL].values
    y = sub[target].values

    zx = robust_z(x)
    zy = robust_z(y)

    # perpendicular distance to y=x in z-space
    perp_dist = np.abs(zy - zx) / rt2
    keep_mask = perp_dist <= PERP_EPS
    kept = sub[keep_mask].copy()

    n_before = len(sub)
    n_kept = len(kept)
    print(f"{target}: kept {n_kept}/{n_before} within PERP_EPS={PERP_EPS}")

    if n_kept < 10:
        print(f"[WARN] Too few after 45° filter for {target}; skipping.")
        continue

    # Plot: z-space (diagnostic)
    zplot = os.path.join(PLOT_DIR, f"{target}_zspace_y_eq_x.png")
    plot_zspace_with_y_eq_x(zx[keep_mask], zy[keep_mask],
                            title=f"{target}: kept points in z-space (near y=x)",
                            outpath=zplot)

    # Plot + LR in original space
    X_kept = kept[[X_COL]].values
    y_kept = kept[target].values
    viz_model = LinearRegression().fit(X_kept, y_kept)

    oplot = os.path.join(PLOT_DIR, f"{target}_origspace_kept.png")
    plot_scatter_with_lr(
        kept[X_COL].values,
        kept[target].values,
        title=f"{target} vs {X_COL} (kept near 45° in z-space)",
        xlabel=X_COL, ylabel=target,
        outpath=oplot,
        model=viz_model
    )

    # Train/test & metrics (original units)
    X_train, X_test, y_train, y_test = train_test_split(
        X_kept, y_kept, test_size=0.2, random_state=42
    )
    final_model = LinearRegression().fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics.append({
        "target": target,
        "n_before": n_before,
        "n_kept": n_kept,
        "PERP_EPS": PERP_EPS,
        "coef_wind_max_m_s": float(final_model.coef_[0]),
        "intercept": float(final_model.intercept_),
        "RMSE": rmse,
        "R2": r2,
        "zspace_plot": zplot,
        "orig_plot": oplot
    })

# ----------------------------
# Print results
# ----------------------------
if metrics:
    mdf = pd.DataFrame(metrics)
    print("\n=== LR after 45°-line (y=x) proximity filter (robust z-space) ===")
    print(mdf[[
        "target","n_before","n_kept","PERP_EPS",
        "coef_wind_max_m_s","intercept","RMSE","R2"
    ]].to_string(index=False))

    print("\nPlots saved in:", os.path.abspath(PLOT_DIR))
    for row in metrics:
        print(f"- {row['target']} z-space: {row['zspace_plot']}")
        print(f"  original: {row['orig_plot']}")
else:
    print("No models trained—45° proximity filter left too few points.")
