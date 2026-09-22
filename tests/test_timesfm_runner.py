"""Focused contract tests for the TimesFM 3.0 adapter."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from demand_forecast import timesfm_runner as runner


class _FakeForecaster:
    def __init__(self) -> None:
        self.context: np.ndarray | None = None
        self.kwargs: dict[str, object] | None = None

    def predict(self, context: np.ndarray, **kwargs: object) -> SimpleNamespace:
        self.context = context
        self.kwargs = kwargs
        horizon = int(kwargs["horizon"])
        quantiles = np.tile(np.arange(1, 10, dtype=np.float32), (horizon, 1))
        return SimpleNamespace(
            forecast=np.full(horizon, 5.0, dtype=np.float32),
            quantiles=quantiles,
        )


class TimesFMRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        runner._MODEL_CACHE = None

    def tearDown(self) -> None:
        runner._MODEL_CACHE = None

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False)
    @patch.object(runner.torch.cuda, "is_available", return_value=True)
    @patch.object(runner.TimesFM3Forecaster, "from_pretrained")
    def test_maps_timesfm3_deciles_and_caps_context(
        self, from_pretrained, _cuda_available
    ) -> None:
        fake = _FakeForecaster()
        from_pretrained.return_value = fake

        forecast = runner.forecast_timesfm(
            np.arange(20_000, dtype=np.float32), h=3, max_context=20_000
        )

        from_pretrained.assert_called_once_with(
            runner.TIMESFM3_CHECKPOINT,
            device="cuda",
            token="test-token",
            per_core_batch_size=4,
        )
        self.assertEqual(len(fake.context), runner.MAX_CONTEXT)
        self.assertEqual(fake.context[0], 4_640)
        self.assertEqual(fake.kwargs, {
            "horizon": 3,
            "return_quantiles": True,
            "make_positive": True,
        })
        self.assertEqual(forecast.name, runner.TIMESFM3_CANDIDATE_NAME)
        np.testing.assert_array_equal(forecast.point, [5.0, 5.0, 5.0])
        np.testing.assert_array_equal(forecast.lower, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(forecast.upper, [9.0, 9.0, 9.0])

    @patch.object(runner.torch.cuda, "is_available", return_value=False)
    def test_rejects_cpu_before_loading_weights(self, _cuda_available) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            runner.get_timesfm_model()


if __name__ == "__main__":
    unittest.main()
