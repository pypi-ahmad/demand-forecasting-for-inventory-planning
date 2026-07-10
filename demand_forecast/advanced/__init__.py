"""Advanced demand-forecasting techniques (v3) — keeps v1/v2 notebooks untouched.

Implements hierarchical forecasting, calendar/promo features, asymmetric
inventory cost, full rolling backtest, smart ensembles, log-stabilized HW,
SARIMAX with exogenous regressors, and quantile-based reorder signals.
"""

from demand_forecast.advanced.pipeline import AdvancedRunResult, run_advanced_pipeline

__all__ = ["AdvancedRunResult", "run_advanced_pipeline"]
