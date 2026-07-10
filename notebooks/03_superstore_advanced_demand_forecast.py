# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: demand-forecast-project
#     language: python
#     name: demand-forecast-project
# ---

# %% [markdown]
# # Advanced Demand Forecasting Tutorial — Superstore (v3)
#
# **This notebook is additive.** Notebooks `01` / `02` and their v1/v2 results stay
# untouched for learning history. Here we implement the **next layer** of
# production techniques and explain *why* each step improves inventory-facing
# forecasts.
#
# ## Learning goals
#
# | Technique | Why it helps |
# |-----------|--------------|
# | Hierarchical (Category × Region → total) | Totals mix different seasons; bottom-up captures structure |
# | Calendar / holiday / promo features | Year-end peaks are partly *explained*, not guessed |
# | log1p + seasonal model | Stabilizes variance that grows with demand level |
# | SARIMAX + exogenous regressors | Classical model + drivers |
# | ML lags (HistGradientBoosting) | Nonlinear lag/promo interactions |
# | TimesFM zero-shot | Foundation prior; peer model (XReg optional via jax) |
# | Smart 2–3 model stack | Diversity without diluting the champion |
# | Asymmetric inventory cost | Stockouts cost more than overstock |
# | Quantile order (SL≈0.9) | Reorder from predictive band, not only the median |
# | Longer rolling-origin | Gate models across many cuts, not one holiday window |
#
# **Baseline to beat (v2 production on same Superstore weekly holdout):**  
# Holt–Winters multiplicative m=52 → MAE **48.1**, MAPE **11.4%**, MASE **0.80**.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

try:
    from IPython import get_ipython
    from IPython.display import display

    IN_IPYTHON = get_ipython() is not None
except ImportError:
    IN_IPYTHON = False

    def display(obj):
        print(obj)

if not IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (11, 4)

_ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from demand_forecast.advanced import run_advanced_pipeline
from demand_forecast.advanced.evaluation import summarize_rolling
from demand_forecast.advanced.features import aggregate_promo_proxy
from demand_forecast.metrics import forecast_metrics


def show_plot() -> None:
    plt.tight_layout()
    if IN_IPYTHON:
        plt.show()
    else:
        plt.show(block=False)
        plt.close("all")


print("Python/path OK; project root:", _ROOT)

# %% [markdown]
# ## 1. Data — same Superstore source as notebook 01
#
# We deliberately reuse the same URL and cleaning so advanced results are
# comparable to v1/v2, not a different dataset in disguise.

# %%
URL = (
    "https://raw.githubusercontent.com/yajasarora/"
    "Superstore-Sales-Analysis-with-Tableau/master/"
    "Superstore%20sales%20dataset.csv"
)
df = pd.read_csv(URL, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Discount"] = pd.to_numeric(df.get("Discount", 0), errors="coerce").fillna(0.0)
df = df.dropna(subset=["Order Date", "Quantity"])
df = df[df["Quantity"] > 0].copy()
print(df.shape, df["Order Date"].min(), "→", df["Order Date"].max())

# Total weekly units (same grain as v1/v2)
y = (
    df.set_index("Order Date")["Quantity"]
    .resample("W-SUN")
    .sum()
    .astype(float)
    .sort_index()
)
y = y.loc[y.ne(0).idxmax() : y.ne(0)[::-1].idxmax()]
print("Weekly total series:", len(y), y.index.min().date(), "→", y.index.max().date())

promo = aggregate_promo_proxy(df, date_col="Order Date", value_col="Discount")
promo = promo.reindex(y.index).fillna(0.0)

# %% [markdown]
# ## 2. Why hierarchy?
#
# A national total is the **sum of many behaviors** (Furniture vs Technology,
# West vs South). Modeling bottoms (Category × Region) then **summing**
# (bottom-up reconciliation) often improves the total because seasonal shape
# differs by segment.
#
# ```text
# Category×Region series  →  HW forecasts  →  sum  →  total demand
# ```

# %% [markdown]
# ## 3. Run the advanced pipeline
#
# The pipeline trains several models on the **same** train window, scores
# holdout MAE/MASE **and** inventory asymmetric cost (underage cost 4× overage),
# builds a small smart stack, and runs rolling-origin checks.

# %%
result = run_advanced_pipeline(
    y,
    dataset="superstore",
    transactions=df,
    date_col="Order Date",
    qty_col="Quantity",
    hierarchy_cols=["Category", "Region"],
    promo_series=promo,
    country="US",
    h=8,
    n_rolling=6,
    underage_cost=4.0,
    overage_cost=1.0,
    seed=SEED,
)

print("=== Pipeline notes ===")
for n in result.notes:
    print(" -", n)

print("\n=== Accuracy leaderboard (MASE sorted) ===")
display(result.leaderboard.round(4))

print("\n=== Inventory cost table (lower asymmetric_cost is better for ops) ===")
display(result.inventory_table.round(2))

print(f"\n>>> ADVANCED CHAMPION (by inventory cost): {result.champion_name}")

# %% [markdown]
# ## 4. Compare to frozen v2 baseline
#
# v2 production champion on Superstore was **HW mul m=52** with MAE≈48.1.
# We recompute that reference metric on this run’s holdout for a clean delta.

# %%
from demand_forecast.advanced.models_exog import forecast_hw_mul

H = result.h
y_train, y_test = result.y_train, result.y_test
v2_ref = forecast_hw_mul(y_train, H, period=52)
v2_metrics = forecast_metrics(y_test.values, v2_ref.point, y_train.values, mase_period=52)
adv_metrics = result.leaderboard.loc[result.champion_name]

compare = pd.DataFrame(
    {
        "v2_hw_mul_m52_reference": v2_metrics,
        f"v3_champion_{result.champion_name}": {
            "MAE": float(adv_metrics["MAE"]),
            "RMSE": float(adv_metrics["RMSE"]),
            "MAPE": float(adv_metrics["MAPE"]),
            "sMAPE": float(adv_metrics.get("sMAPE", np.nan)),
            "MASE": float(adv_metrics["MASE"]),
            "bias": float(adv_metrics["bias"]),
        },
    }
).T
# also show best MASE model in v3 if different from inventory champion
best_mase_name = result.leaderboard.index[0]
if best_mase_name != result.champion_name:
    compare.loc[f"v3_best_mase_{best_mase_name}"] = result.leaderboard.loc[best_mase_name]

print("=== v2 reference vs v3 ===")
display(compare.round(4))

if "hierarchy_bottom_up" in result.forecasts:
    hm = forecast_metrics(
        y_test.values,
        result.forecasts["hierarchy_bottom_up"].point,
        y_train.values,
        mase_period=52,
    )
    print("Hierarchy bottom-up alone:", {k: round(v, 4) for k, v in hm.items()})

# %% [markdown]
# ## 5. Plots — champion vs actual vs v2 reference

# %%
idx = y_test.index
fig, ax = plt.subplots(figsize=(11, 4.5))
y_train.iloc[-40:].plot(ax=ax, label="train tail", color="steelblue", alpha=0.7)
y_test.plot(ax=ax, label="actual", color="black", lw=2.5)
pd.Series(v2_ref.point, index=idx).plot(ax=ax, label="v2 HW mul m=52", color="gray", ls="--")
pd.Series(result.champion_point, index=idx).plot(
    ax=ax, label=f"v3 champion: {result.champion_name}", color="darkorange", lw=2
)
ax.fill_between(
    idx,
    result.champion_lower,
    result.champion_upper,
    color="darkorange",
    alpha=0.2,
    label="v3 PI band",
)
pd.Series(result.order_qty_sl90, index=idx).plot(
    ax=ax, label="SL≈0.9 order qty", color="green", lw=1.5
)
ax.set_title("Superstore advanced: actual vs v2 baseline vs v3 champion")
ax.set_ylabel("Units / week")
ax.legend()
show_plot()

# %% [markdown]
# ## 6. Rolling-origin robustness
#
# A single holdout can flatter a model. Rolling origin re-fits across multiple
# cut dates and reports mean MAE / MASE / asymmetric cost.

# %%
for name, rdf in result.rolling_summary.items():
    print(f"\n--- rolling: {name} ---")
    if "error" in rdf.columns and rdf["error"].notna().all():
        print(rdf)
        continue
    display(summarize_rolling(rdf).round(3))

# %% [markdown]
# ## 7. Inventory interpretation
#
# - **Point forecast** → baseline replenishment  
# - **Upper PI / SL 0.9 order qty** → service-minded intake  
# - **Safety buffer** ≈ upper − point  
# - **Asymmetric cost** with underage=4, overage=1 approximates “stockouts hurt more”

# %%
print("Champion:", result.champion_name)
print("Holdout actual mean:", float(y_test.mean()))
print("Point mean:", float(np.mean(result.champion_point)))
print("SL0.9 order mean:", float(np.mean(result.order_qty_sl90)))
print("Mean safety buffer:", float(np.mean(result.safety_buffer)))
print(
    result.inventory_table.loc[result.champion_name][
        ["under_units", "over_units", "asymmetric_cost", "asymmetric_cost_sl90_order"]
    ]
)

# Persist
out_dir = _ROOT / "data" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(out_dir / "superstore_v3_accuracy.csv")
result.inventory_table.to_csv(out_dir / "superstore_v3_inventory.csv")
compare.to_csv(out_dir / "superstore_v2_vs_v3.csv")
print("Wrote CSVs to", out_dir)

# %% [markdown]
# ## 8. Takeaways for learners
#
# 1. **v1** taught PyCaret survey + TimesFM comparison.  
# 2. **v2** fixed seasonality period / multiplicative HW → large accuracy jump.  
# 3. **v3 (this notebook)** adds hierarchy, drivers, inventory loss, smart stacks,
#    and longer rolling tests — closer to how a demand-planning team would gate models.
#
# If v3 does not beat v2 MAE on this short peak holdout, check **asymmetric cost**
# and **rolling mean** — those are the production KPIs we optimized for.
#
# Notebook 03 complete.
print("Notebook 03 complete.")
