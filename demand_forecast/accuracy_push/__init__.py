"""Accuracy-push layer (v4): beat v2 univariate HW via hierarchy + multi-window selection.

Does not modify v1–v3 notebooks; used by notebooks 05/06.
"""

from demand_forecast.accuracy_push.pipeline import AccuracyPushResult, run_accuracy_push

__all__ = ["AccuracyPushResult", "run_accuracy_push"]
