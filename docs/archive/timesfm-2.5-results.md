# Historical TimesFM 2.5 results

This document preserves the TimesFM 2.5 benchmark record that was current
before the TimesFM 3.0 migration on 2026-09-22. It is an audit artifact, not a
benchmark for the current runtime.

The source code, notebook outputs, and README tables that produced these
figures are available at commit
[`945ae21`](https://github.com/pypi-ahmad/demand-forecasting-for-inventory-planning/tree/945ae21f00b70ecf5cecc709b6dc15ff585e28b7).
That revision used `google/timesfm-2.5-200m-pytorch` through the older
`TimesFM_2p5_200M_torch` API on the same RTX 4060 workstation.

## What these results mean

- They are zero-shot TimesFM 2.5 results on the historical Superstore and
  Online Retail II notebook runs.
- They used the datasets, holdouts, classical candidates, and metric settings
  captured in the linked revision.
- They must not be compared directly with TimesFM 3.0 or described as TimesFM
  3.0 performance.
- The active notebooks now contain independently executed TimesFM 3.0 results.
  They are not directly comparable to these historical numbers.

## Preserved headline metrics

| Dataset | Historical model | MAE | RMSE | MAPE | MASE |
|---|---|---:|---:|---:|---:|
| Superstore | TimesFM 2.5 zero-shot | 78.84 | 93.50 | 18.39% | 1.316 |
| Online Retail II | TimesFM 2.5 zero-shot | 31,732 | 40,347 | 16.88% | 0.793 |

The linked source revision remains the authoritative record for the full
leaderboards, rolling-origin outputs, figures, and later pipeline generations.
Keeping the detailed artifacts there avoids copying old execution output into
the current TimesFM 3.0 tutorial.

## License boundary

The archived model path and the current TimesFM 3.0 path have different weight
terms. Consult the applicable upstream model card before using either set of
weights. The TimesFM 3.0 weight restriction is documented in the active
[README](../../README.md).
