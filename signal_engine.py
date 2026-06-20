"""
signal_engine.py
Rolling PCA residual flow alpha engine.
  1. Rolling PCA on TFI matrix (120-bar window)
  2. Rolling OLS residualization (60-bar beta window)
  3. Cross-sectional z-score of residuals
  4. Blend with funding signal (0.8 / 0.2)
  5. Apply DirectionSign (from IC validation)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, zscore as scipy_zscore


PCA_WINDOW  = 120   # bars for PCA covariance estimation
BETA_WINDOW = 60    # bars for OLS beta estimation
N_COMPONENTS = 2    # PC1 + PC2


# ── Rolling PCA PC scores (strict lookahead protection) ──────────────────────

def compute_rolling_pc_scores(tfi_matrix: pd.DataFrame,
                               pca_window: int = PCA_WINDOW,
                               n_components: int = N_COMPONENTS) -> pd.DataFrame:
    """
    For each timestamp t, fit PCA only on [t-pca_window, t-1],
    then project X_t to get PC scores.
    Returns DataFrame of shape (len(tfi_matrix), n_components).
    """
    n = len(tfi_matrix)
    if n < pca_window + 1:
        return pd.DataFrame()

    pc_scores = []
    prev_comps = None
    for t in range(pca_window, n):
        window_data = tfi_matrix.iloc[t - pca_window : t].values  # past only
        current     = tfi_matrix.iloc[t].values.reshape(1, -1)

        scaler = StandardScaler()
        w_scaled = scaler.fit_transform(window_data)
        cur_scaled = scaler.transform(current)

        pca = PCA(n_components=n_components)
        pca.fit(w_scaled)
        comps = pca.components_[:n_components]
        # Sign-pin PCs across rolling windows: flip each component to align with
        # the previous window, and apply the same flip to its projected score.
        # Without this, sklearn's per-fit svd_flip lets PC2's sign flip between
        # adjacent refits, inverting the residual sign for PC2-loaded assets.
        flips = _component_sign_flips(comps, prev_comps)
        prev_comps = comps * flips[:, None]
        scores = pca.transform(cur_scaled)[0] * flips
        pc_scores.append(scores)

    idx = tfi_matrix.index[pca_window:]
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(pc_scores, index=idx, columns=cols)


def _component_sign_flips(comps: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """Per-component ±1 sign flips to orient PCs consistently across windows.

    Aligns to the previous window by max overlap; on the first window anchors the
    largest-magnitude loading positive (deterministic, sklearn svd_flip style).
    """
    flips = np.ones(comps.shape[0])
    for k in range(comps.shape[0]):
        if prev is not None and k < prev.shape[0]:
            if float(np.dot(comps[k], prev[k])) < 0.0:
                flips[k] = -1.0
        else:
            j = int(np.argmax(np.abs(comps[k])))
            if comps[k][j] < 0.0:
                flips[k] = -1.0
    return flips


def get_pc1_explained_variance(tfi_matrix: pd.DataFrame,
                                pca_window: int = PCA_WINDOW) -> float:
    """
    Compute current PC1 explained variance ratio using the latest window.
    """
    if len(tfi_matrix) < pca_window:
        return 0.0
    window_data = tfi_matrix.tail(pca_window).values
    scaler = StandardScaler()
    w_scaled = scaler.fit_transform(window_data)
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(w_scaled)
    return float(pca.explained_variance_ratio_[0])


# ── Rolling OLS residualization ───────────────────────────────────────────────

def compute_residuals(tfi_matrix: pd.DataFrame,
                      pc_scores: pd.DataFrame,
                      beta_window: int = BETA_WINDOW) -> pd.DataFrame:
    """
    For each asset and each time t, regress TFI on PC1+PC2 using
    the past beta_window observations, then compute OOS residual at t.
    Returns cross-sectional z-scored residuals.
    """
    # Align
    common_idx = tfi_matrix.index.intersection(pc_scores.index)
    tfi = tfi_matrix.loc[common_idx]
    pcs = pc_scores.loc[common_idx]

    n = len(tfi)
    if n < beta_window + 1:
        return pd.DataFrame()

    residuals = {sym: [] for sym in tfi.columns}
    valid_idx = []

    for t in range(beta_window, n):
        tfi_win = tfi.iloc[t - beta_window : t].values          # (beta_window, N)
        pc_win  = pcs.iloc[t - beta_window : t].values          # (beta_window, 2)
        X = np.column_stack([np.ones(beta_window), pc_win])     # add intercept

        tfi_cur = tfi.iloc[t].values
        pc_cur  = pcs.iloc[t].values
        x_cur   = np.array([1.0, pc_cur[0], pc_cur[1]])

        row_resid = []
        for j in range(tfi_win.shape[1]):
            y = tfi_win[:, j]
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                pred = x_cur @ beta
                row_resid.append(tfi_cur[j] - pred)
            except Exception:
                row_resid.append(0.0)

        row_arr = np.array(row_resid, dtype=float)
        # Cross-sectional z-score
        mu, sigma = row_arr.mean(), row_arr.std(ddof=1)
        if sigma > 1e-10:
            row_z = (row_arr - mu) / sigma
        else:
            row_z = np.zeros_like(row_arr)

        for j, sym in enumerate(tfi.columns):
            residuals[sym].append(row_z[j])
        valid_idx.append(tfi.index[t])

    return pd.DataFrame(residuals, index=valid_idx)


# ── Alpha construction ────────────────────────────────────────────────────────

def compute_raw_alpha(residual_z: pd.Series,
                      funding_signals: dict,
                      alpha_w: float = 0.8) -> dict:
    """
    Alpha_i = 0.8 * ResidualZ_i + 0.2 * FundingSignal_i
    Returns dict {symbol: alpha_value}
    """
    result = {}
    for sym, rz in residual_z.items():
        fs = funding_signals.get(sym, 0.0)
        result[sym] = alpha_w * rz + (1 - alpha_w) * fs
    return result


# ── IC computation (for direction validation) ─────────────────────────────────

def compute_ic(alpha_series: pd.DataFrame,
               return_series: pd.DataFrame,
               horizon: int = 1) -> float:
    """
    IC(h) = mean over t of cross-sectional SpearmanCorr(Alpha_t, r_{t+h}).

    Per-period (cross-sectional) IC averaged across time — NOT a single pooled
    rank correlation over a flattened (time x asset) array. Pooling would inflate
    the sample size by the number of assets, understate the standard error, and
    conflate cross-sectional skill with time-series autocorrelation. Here each
    period contributes one cross-sectional IC and those are averaged (the
    Fama-MacBeth-style estimator the strategy spec intends).

    alpha_series and return_series must be column-aligned DataFrames (one column
    per asset), positionally paired so alpha row t predicts return row t+horizon.
    """
    if len(alpha_series) < horizon + 10:
        return 0.0

    alpha = alpha_series.iloc[:-horizon]
    rets = return_series.iloc[horizon:]
    cols = alpha.columns.intersection(rets.columns)
    if len(cols) < 3:  # need a cross-section to rank-correlate
        return 0.0
    alpha = alpha[cols].to_numpy(dtype=float)
    rets = rets[cols].to_numpy(dtype=float)

    period_ics = []
    for av, rv in zip(alpha, rets):
        valid = ~(np.isnan(av) | np.isnan(rv))
        if valid.sum() < 3:
            continue
        a, r = av[valid], rv[valid]
        if np.std(a) < 1e-12 or np.std(r) < 1e-12:
            continue
        corr, _ = spearmanr(a, r)
        if not np.isnan(corr):
            period_ics.append(corr)

    if not period_ics:
        return 0.0
    return float(np.mean(period_ics))


# ── Direction sign ────────────────────────────────────────────────────────────

def get_direction_sign(ic_at_h1: float) -> int:
    """Returns +1 for continuation, -1 for reversal."""
    return 1 if ic_at_h1 >= 0 else -1


# ── Main signal update (called every minute) ──────────────────────────────────

def compute_current_alpha(tfi_matrix: pd.DataFrame,
                          funding_signals: dict,
                          direction_sign: int = 1,
                          pca_window: int = PCA_WINDOW,
                          beta_window: int = BETA_WINDOW,
                          alpha_w: float = 0.8) -> dict:
    """
    Full pipeline for a single update:
      TFI matrix → PC scores → residuals → blend funding → apply direction.
    Returns:
      {
        "adjusted_alpha": {sym: value},
        "raw_alpha":       {sym: value},
        "residual_z":      {sym: value},
        "pc1_var":         float,
        "pc_scores":       pd.DataFrame,
      }
    """
    if tfi_matrix is None or len(tfi_matrix) < pca_window + beta_window + 2:
        return {"adjusted_alpha": {}, "raw_alpha": {},
                "residual_z": {}, "pc1_var": 0.0, "pc_scores": pd.DataFrame()}

    # Only use non-BTC columns
    trade_cols = [c for c in tfi_matrix.columns if c != "BTCUSDT"]
    all_cols   = list(tfi_matrix.columns)  # BTC included for PCA factor extraction

    pc_scores = compute_rolling_pc_scores(tfi_matrix[all_cols], pca_window, N_COMPONENTS)
    if pc_scores.empty:
        return {"adjusted_alpha": {}, "raw_alpha": {},
                "residual_z": {}, "pc1_var": 0.0, "pc_scores": pc_scores}

    residuals = compute_residuals(tfi_matrix[trade_cols], pc_scores, beta_window)
    if residuals.empty:
        return {"adjusted_alpha": {}, "raw_alpha": {},
                "residual_z": {}, "pc1_var": 0.0, "pc_scores": pc_scores}

    # Latest residual z-scores
    latest_resid = residuals.iloc[-1].to_dict()

    # Raw alpha
    raw_alpha = compute_raw_alpha(latest_resid, funding_signals, alpha_w)

    # Adjusted alpha
    adj_alpha = {sym: direction_sign * v for sym, v in raw_alpha.items()}

    pc1_var = get_pc1_explained_variance(tfi_matrix[all_cols], pca_window)

    return {
        "adjusted_alpha": adj_alpha,
        "raw_alpha":       raw_alpha,
        "residual_z":      latest_resid,
        "pc1_var":         pc1_var,
        "pc_scores":       pc_scores,
        "residuals_history": residuals,
    }
