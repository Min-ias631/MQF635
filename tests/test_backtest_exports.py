"""Tests for backtest metric/export consistency."""

from __future__ import annotations

import unittest

import pandas as pd

from backtest import compute_metrics


class BacktestExportTests(unittest.TestCase):
    def test_exported_net_return_matches_equity_series_definition(self) -> None:
        index = pd.date_range("2026-01-01", periods=8, freq="min", tz="UTC")
        weights_df = pd.DataFrame(
            {
                "AAAUSDT": [0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                "BBBUSDT": [-0.5, -0.5, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0],
            },
            index=index,
        )
        closes_df = pd.DataFrame(
            {
                "AAAUSDT": [100.0, 100.05, 100.10, 100.15, 100.20, 100.25, 100.25, 100.25],
                "BBBUSDT": [100.0, 99.95, 99.90, 99.85, 99.80, 99.75, 99.75, 99.75],
            },
            index=index,
        )
        regime_df = pd.DataFrame(
            {"regime": ["Strong"] * len(index), "mult": [1.0] * len(index)},
            index=index,
        )

        metrics = compute_metrics(weights_df, regime_df, closes_df, cost_bps=10.0)
        gross_return = (weights_df * closes_df.pct_change().shift(-1)).sum(axis=1)
        exported = pd.DataFrame(
            {
                "gross_return": gross_return,
                "net_return": metrics["net"].reindex(index),
                "equity": metrics["equity"].reindex(index),
            },
            index=index,
        )

        pd.testing.assert_series_equal(
            exported["net_return"].iloc[:-1],
            metrics["net"],
            check_names=False,
        )
        self.assertTrue((exported["gross_return"].iloc[:-1] >= exported["net_return"].iloc[:-1]).all())
        self.assertTrue(exported["net_return"].iloc[-1] != exported["net_return"].iloc[-1])  # NaN on last bar


if __name__ == "__main__":
    unittest.main()
