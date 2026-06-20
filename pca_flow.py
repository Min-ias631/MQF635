"""pca_flow.py — rolling PCA residual-flow alpha (canonical engine).

Direct port of ``eugene-ui/strategy/signal_engine.py`` using numpy only (no
sklearn/scipy), so the engine has the same dependency surface as the Yumi
backtest. The pipeline per update:

  1. Append the current cross-sectional TFI vector to a rolling buffer
     (one row per minute bar; intra-minute re-calls replace the live row).
  2. Rolling PCA on the TFI matrix (fit on the past ``pca_window`` rows,
     project the current row) -> PC1+PC2 scores.
  3. Rolling OLS residualization of each TRADE symbol's TFI on PC1+PC2 over
     the past ``beta_window`` rows -> out-of-sample residual.
  4. Cross-sectional z-score of residuals.
  5. Blend with the funding signal: flow_weight*z + funding_weight*funding.
  6. Apply ``direction_sign``.

PCA columns include BTC (factor extraction); residualization/alpha is produced
only for non-BTC trade symbols, matching the original ``compute_current_alpha``.

Each rolling window refits PCA independently; PC orientations are sign-pinned
across windows (see ``_sign_align``) so PC1/PC2 cannot flip sign between adjacent
refits — an unpinned flip would invert the OLS beta/residual (and thus the alpha
sign) for assets loading on that component.

Direction: when ``AlphaConfig.ic_stability_gate`` is on and enough finalized
minute bars have accumulated, DirectionSign is set live from out-of-sample IC
(per-period cross-sectional Spearman, t-stat and cross-horizon sign-stability
checks); when the gate decides the signal is unreliable it returns 0 and the
adjusted alpha is zeroed (no new positions). Until then the configured
``direction_sign`` prior is used.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class PCAFlowAlpha:
    def __init__(self, pca_symbols, config) -> None:
        self.pca_symbols: Tuple[str, ...] = tuple(s.upper() for s in pca_symbols)
        # Trade universe = PCA universe minus BTC (the market factor), matching
        # signal_engine's `trade_cols = [c for c in cols if c != "BTCUSDT"]`.
        self.trade_symbols: Tuple[str, ...] = tuple(
            s for s in self.pca_symbols if s != "BTCUSDT"
        )
        self._trade_idx = [self.pca_symbols.index(s) for s in self.trade_symbols]
        self.config = config

        self._buffer: List[np.ndarray] = []          # rows of TFI over pca_symbols
        self._last_minute: Optional[Any] = None       # minute key of the live row
        self._latest_funding_rates: Dict[str, float] = {}
        self._cache: Tuple[Dict[str, float], Dict[str, float]] = ({}, {})

        # IC stability gate state: finalized per-minute (raw alpha vec, close vec)
        # pairs feed the out-of-sample IC estimate that sets DirectionSign live.
        self._finalized: Deque[Tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=int(getattr(config, "ic_lookback", 240))
        )
        self._pending: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._gate_sign: int = 0
        self._gate_ready: bool = False
        self._gate_stats: Dict[str, Any] = {}
        self._smoothed_alpha: Dict[str, float] = {}   # EWMA state for turnover control

    # ------------------------------------------------------------------ public

    def update(self, snapshot) -> Dict[str, float]:
        """Ingest a snapshot, return adjusted alpha per trade symbol."""
        row = np.array(
            [self._bar_tfi(snapshot.bars.get(s)) for s in self.pca_symbols],
            dtype=float,
        )
        minute_key = self._minute_key(snapshot.timestamp)
        if self._buffer and self._last_minute == minute_key:
            self._buffer[-1] = row            # same bar still forming -> replace
        else:
            # New minute: the previous minute's alpha/close pair is now final and
            # can join the IC history (its forward return becomes observable).
            if self._pending is not None:
                self._finalized.append(self._pending)
                self._pending = None
                self._refresh_gate()
            self._buffer.append(row)
            self._last_minute = minute_key

        max_len = self.config.pca_window + self.config.beta_window + 5
        if len(self._buffer) > max_len:
            self._buffer = self._buffer[-max_len:]

        self._latest_funding_rates = {
            s: float(snapshot.funding[s].rate) if s in snapshot.funding else 0.0
            for s in self.trade_symbols
        }

        adjusted, raw_alpha, residual_z, funding_signal = self._compute()
        self._cache = (residual_z, funding_signal)

        if raw_alpha:
            closes = np.array(
                [
                    float(snapshot.bars[s].close) if s in snapshot.bars else np.nan
                    for s in self.trade_symbols
                ],
                dtype=float,
            )
            alpha_vec = np.array(
                [raw_alpha.get(s, np.nan) for s in self.trade_symbols], dtype=float
            )
            self._pending = (alpha_vec, closes)

        return self._smooth(adjusted)

    def _smooth(self, adjusted: Dict[str, float]) -> Dict[str, float]:
        """EWMA the traded alpha over smooth_span bars to stabilize Q80/Q20 churn."""
        span = int(getattr(self.config, "smooth_span", 1) or 1)
        if span <= 1 or not adjusted:
            return adjusted
        a = 2.0 / (span + 1.0)
        out = {}
        for s in self.trade_symbols:
            new = float(adjusted.get(s, 0.0))
            prev = self._smoothed_alpha.get(s, new)
            out[s] = a * new + (1.0 - a) * prev
        self._smoothed_alpha = out
        return out

    def raw_components(self, snapshot=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (residual_z, funding_signal) cached from the last update.

        ``update`` is always invoked immediately before this with the same
        snapshot, so we return the cache instead of recomputing.
        """
        return self._cache

    @property
    def direction_info(self) -> Dict[str, Any]:
        """Diagnostics for the IC direction gate (for dashboards/logging)."""
        gate_on = bool(getattr(self.config, "ic_stability_gate", False))
        return {
            "gate_enabled": gate_on,
            "active": bool(gate_on and self._gate_ready),
            "direction": self._effective_sign(),
            "fallback_sign": int(self.config.direction_sign),
            "periods": len(self._finalized),
            **self._gate_stats,
        }

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _bar_tfi(bar) -> float:
        if bar is None:
            return 0.0
        return float(bar.tfi)

    @staticmethod
    def _minute_key(ts):
        # ts is a datetime; bucket to the minute so multiple intra-minute
        # updates collapse onto one bar row.
        try:
            return ts.replace(second=0, microsecond=0)
        except AttributeError:
            return ts

    def _compute(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        n = len(self._buffer)
        pca_window = self.config.pca_window
        beta_window = self.config.beta_window
        if n < pca_window + beta_window + 1:
            return {}, {}, {}, {}

        x_full = np.vstack(self._buffer)                  # (n, k)
        pc = self._rolling_pc_scores(x_full)              # (n - pca_window, nc)
        if pc is None or pc.shape[0] < beta_window + 1:
            return {}, {}, {}, {}

        x_trade = x_full[:, self._trade_idx]              # (n, k_trade)
        x_trade_aligned = x_trade[pca_window:]            # align to pc rows
        z = self._latest_residual_z(x_trade_aligned, pc)
        if z is None:
            return {}, {}, {}, {}

        residual_z = {s: float(z[i]) for i, s in enumerate(self.trade_symbols)}
        funding_signal = self._funding_signal()

        fw = self.config.flow_weight
        gw = self.config.funding_weight
        raw_alpha = {
            s: float(fw * residual_z[s] + gw * funding_signal.get(s, 0.0))
            for s in self.trade_symbols
        }
        sign = self._effective_sign()
        adjusted = {s: float(sign * a) for s, a in raw_alpha.items()}
        return adjusted, raw_alpha, residual_z, funding_signal

    # -------------------------------------------------------- IC direction gate

    def _effective_sign(self) -> int:
        """DirectionSign: live IC gate when ready, else the configured prior."""
        if getattr(self.config, "ic_stability_gate", False) and self._gate_ready:
            return self._gate_sign
        return int(self.config.direction_sign)

    def _refresh_gate(self) -> None:
        """Recompute the IC direction gate from finalized (alpha, close) history.

        Per strategy.md: DirectionSign = +1/-1 only if IC(1) clears ic_min with a
        t-stat beyond ic_tstat_min and the mean IC sign agrees across ic_horizons;
        otherwise 0 (signal unreliable, open no new positions).
        """
        cfg = self.config
        if not getattr(cfg, "ic_stability_gate", False):
            return
        n = len(self._finalized)
        if n < int(cfg.ic_min_periods):
            self._gate_ready = False
            return

        alphas = np.vstack([a for a, _ in self._finalized])   # (n, k)
        closes = np.vstack([c for _, c in self._finalized])   # (n, k)

        means: Dict[int, float] = {}
        tstat1 = 0.0
        for h in cfg.ic_horizons:
            ics = self._period_ics(alphas, closes, int(h))
            if len(ics) < 2:
                self._gate_ready = False
                return
            means[int(h)] = float(np.mean(ics))
            if int(h) == 1:
                sd = float(np.std(ics, ddof=1))
                if sd > 1e-12:
                    tstat1 = means[1] / (sd / np.sqrt(len(ics)))
                else:
                    # Zero dispersion: every period agreed exactly. A constant
                    # nonzero IC is maximal evidence, not zero evidence.
                    tstat1 = float(np.sign(means[1])) * float("inf") if abs(means[1]) > 1e-12 else 0.0

        ic1 = means.get(1, 0.0)
        stable = len({1 if m > 0 else -1 for m in means.values()}) == 1
        if ic1 > cfg.ic_min and tstat1 > cfg.ic_tstat_min and stable:
            sign = 1
        elif ic1 < -cfg.ic_min and tstat1 < -cfg.ic_tstat_min and stable:
            sign = -1
        else:
            sign = 0

        self._gate_sign = sign
        self._gate_ready = True
        self._gate_stats = {
            "ic1": round(ic1, 4),
            "tstat": round(tstat1, 2),
            "ic_by_h": {h: round(m, 4) for h, m in means.items()},
            "signs_stable": stable,
        }

    def _period_ics(self, alphas: np.ndarray, closes: np.ndarray, h: int) -> List[float]:
        """Per-period cross-sectional Spearman IC of alpha_t vs r_{t -> t+h}."""
        n = alphas.shape[0]
        ics: List[float] = []
        for t in range(n - h):
            a = alphas[t]
            c0, c1 = closes[t], closes[t + h]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = c1 / c0 - 1.0
            valid = np.isfinite(a) & np.isfinite(r)
            if valid.sum() < 3:
                continue
            av, rv = a[valid], r[valid]
            if np.std(av) < 1e-12 or np.std(rv) < 1e-12:
                continue
            ic = self._spearman(av, rv)
            if np.isfinite(ic):
                ics.append(float(ic))
        return ics

    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = PCAFlowAlpha._rank(a), PCAFlowAlpha._rank(b)
        sa, sb = ra.std(), rb.std()
        if sa < 1e-12 or sb < 1e-12:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    @staticmethod
    def _rank(a: np.ndarray) -> np.ndarray:
        """Average ranks (ties averaged), matching scipy.stats.rankdata."""
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        sa = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sa[j + 1] == sa[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks

    def _rolling_pc_scores(self, x: np.ndarray) -> Optional[np.ndarray]:
        """For each t in [pca_window, n): fit PCA on [t-pca_window, t), project x_t.

        Faithful numpy reimplementation of signal_engine.compute_rolling_pc_scores:
        StandardScaler (population std, zero-variance -> scale 1) + PCA(n_components).
        """
        n = x.shape[0]
        pca_window = self.config.pca_window
        nc = self.config.n_components
        if n < pca_window + 1:
            return None

        scores = []
        prev_comps: Optional[np.ndarray] = None
        for t in range(pca_window, n):
            window = x[t - pca_window:t]                  # past only
            cur = x[t]

            mu = window.mean(axis=0)
            sigma = window.std(axis=0)                    # ddof=0, like StandardScaler
            sigma = np.where(sigma < 1e-12, 1.0, sigma)
            w_scaled = (window - mu) / sigma              # already zero-mean per col
            cur_scaled = (cur - mu) / sigma

            # PCA via SVD on the (already centred) standardized window.
            # Vt rows are ordered by descending singular value == sklearn PCA.
            _, _, vt = np.linalg.svd(w_scaled, full_matrices=False)
            comps = vt[:nc]                               # (nc, k)
            # Sign-pin PCs across rolling windows so PC1/PC2 do not flip sign
            # between adjacent refits. An eigenvector sign flip would otherwise
            # invert the OLS beta (and hence the residual) for assets loading on
            # that component, silently flipping the alpha sign.
            comps = self._sign_align(comps, prev_comps)
            prev_comps = comps
            scores.append(comps @ cur_scaled)            # (nc,)

        return np.asarray(scores)

    @staticmethod
    def _sign_align(comps: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        """Orient PCA components consistently across rolling windows.

        - With a previous window: flip each component whose dot product with the
          previous orientation is negative (max-overlap alignment).
        - First window (no previous): deterministic anchor — make the largest-
          magnitude loading positive (sklearn ``svd_flip`` convention), so the
          whole rolling series is reproducible across calls.
        """
        aligned = comps.copy()
        for k in range(aligned.shape[0]):
            if prev is not None and k < prev.shape[0]:
                if float(np.dot(aligned[k], prev[k])) < 0.0:
                    aligned[k] = -aligned[k]
            else:
                j = int(np.argmax(np.abs(aligned[k])))
                if aligned[k][j] < 0.0:
                    aligned[k] = -aligned[k]
        return aligned

    def _latest_residual_z(
        self, x_trade: np.ndarray, pc: np.ndarray
    ) -> Optional[np.ndarray]:
        """Latest OOS residual z-score, ported from compute_residuals (last row)."""
        beta_window = self.config.beta_window
        m = x_trade.shape[0]
        if m < beta_window + 1:
            return None

        t = m - 1
        tfi_win = x_trade[t - beta_window:t]              # (beta_window, k_trade)
        pc_win = pc[t - beta_window:t]                    # (beta_window, nc)
        x_reg = np.column_stack([np.ones(beta_window), pc_win])  # intercept + PCs

        tfi_cur = x_trade[t]
        x_cur = np.concatenate([[1.0], pc[t]])

        resid = np.empty(x_trade.shape[1], dtype=float)
        for j in range(x_trade.shape[1]):
            try:
                beta, _, _, _ = np.linalg.lstsq(x_reg, tfi_win[:, j], rcond=None)
                resid[j] = tfi_cur[j] - x_cur @ beta
            except Exception:
                resid[j] = 0.0

        mu = resid.mean()
        sigma = resid.std(ddof=1)
        if sigma > 1e-10:
            return (resid - mu) / sigma
        return np.zeros_like(resid)

    def _funding_signal(self) -> Dict[str, float]:
        """Cross-sectional funding signal: -tanh(zscore(rate)).

        Ported from features.compute_funding_signal (scipy zscore -> ddof=0).
        """
        rates = self._latest_funding_rates
        syms = list(rates.keys())
        vals = np.array([rates[s] for s in syms], dtype=float)
        if len(vals) < 2 or np.std(vals) == 0:
            return {s: 0.0 for s in syms}
        z = (vals - vals.mean()) / vals.std()             # ddof=0
        return {s: float(-np.tanh(zi)) for s, zi in zip(syms, z)}


__all__ = ["PCAFlowAlpha"]
