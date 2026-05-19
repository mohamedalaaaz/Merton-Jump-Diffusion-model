"""
eq_futures_jd_v1.py
===================
Merton Jump-Diffusion calibration — CME Equity Index Futures
Supports: E-mini / Micro E-mini NASDAQ 100, S&P 500, Dow Jones

How this differs from btc_jd_futures_v1.py
-------------------------------------------
EQ 1   Data source → yfinance (Yahoo Finance)
        Symbols supported:
          NQ=F   E-mini NASDAQ 100       $20/pt  tick=$5.00
          MNQ=F  Micro E-mini NASDAQ     $2/pt   tick=$0.50   ← default
          ES=F   E-mini S&P 500          $50/pt  tick=$12.50
          MES=F  Micro E-mini S&P 500    $5/pt   tick=$1.25
          YM=F   E-mini Dow Jones        $5/pt   tick=$5.00
          MYM=F  Micro E-mini Dow Jones  $0.50/pt tick=$0.50
        Auxiliary: ^VIX (fear gauge), ^TNX (10Y yield for carry)
        Cross-instrument: all 3 indices fetched for correlation panel

EQ 2   Cost of carry replaces funding rate
        Equity index futures: F = S × e^((r_f − q) × T)
        Daily carry cost ≈ (10Y_yield − div_yield) / 252
        Dividend yield defaults: NASDAQ≈0.8%, S&P≈1.5%, Dow≈2.0%
        r_adj = r_raw − daily_carry
        CLI flag --no-carry skips the adjustment

EQ 3   VIX panel replaces OI panel
        VIX history + rolling 21d VIX percentile
        VIX regime overlay vs HMM-detected regime
        VIX > 30 = high fear (historically bullish ~3-6 months forward)
        VIX < 15 = complacency

EQ 4   Cross-instrument correlation panel
        Rolling 21-day Pearson correlation: NQ vs ES vs YM
        Divergence between indices = early warning signal

EQ 5   Contract specs in position panel
        Tick value, notional at last price, margin estimate
        Dollar P&L per ¼-Kelly lot

Inherited from btc_jd_futures_v1.py
-------------------------------------
FIX 1  Lee-Mykland jump filter (sig=0.001, adaptive window)
FIX 2  λ upper-bound=52, log-space soft penalty
FIX 3  2-state HMM regime detection, per-regime JD, blended Kelly

Usage
-----
    python eq_futures_jd_v1.py                      # MNQ=F default
    python eq_futures_jd_v1.py --symbol ES=F        # Micro S&P
    python eq_futures_jd_v1.py --symbol NQ=F        # Full E-mini NASDAQ
    python eq_futures_jd_v1.py --symbol YM=F        # Dow Jones
    python eq_futures_jd_v1.py --no-carry           # raw returns
    python eq_futures_jd_v1.py --show               # interactive window
    python eq_futures_jd_v1.py --synth              # offline self-test
    python eq_futures_jd_v1.py --days 730

Requirements: numpy scipy pandas matplotlib yfinance
    pip install yfinance
"""

import os
import warnings
import argparse
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
import ssl
import certifi

# ─── Windows SSL fix (compatible with yfinance ≥ 0.2.50 / curl_cffi) ────────
# New yfinance uses curl_cffi internally and manages its own session.
# DO NOT pass session= to yf.download() — it raises YFDataException.
# Fix is env-var only: point curl + Python ssl at certifi's CA bundle.
os.environ["SSL_CERT_FILE"]      = certifi.where()   # used by curl_cffi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()   # used by requests fallback
os.environ["CURL_CA_BUNDLE"]     = certifi.where()   # used by libcurl directly
ssl._create_default_https_context = ssl.create_default_context

import yfinance as yf

# ─── Contract specifications ─────────────────────────────────────────────────

CONTRACT_SPECS = {
    "NQ=F":  {"name": "E-mini NASDAQ 100",       "mult": 20,   "tick": 0.25, "tick_val": 5.00,  "div_yield": 0.008, "margin_est": 22000},
    "MNQ=F": {"name": "Micro E-mini NASDAQ 100", "mult": 2,    "tick": 0.25, "tick_val": 0.50,  "div_yield": 0.008, "margin_est": 2200},
    "ES=F":  {"name": "E-mini S&P 500",          "mult": 50,   "tick": 0.25, "tick_val": 12.50, "div_yield": 0.015, "margin_est": 15000},
    "MES=F": {"name": "Micro E-mini S&P 500",    "mult": 5,    "tick": 0.25, "tick_val": 1.25,  "div_yield": 0.015, "margin_est": 1500},
    "YM=F":  {"name": "E-mini Dow Jones",         "mult": 5,    "tick": 1.00, "tick_val": 5.00,  "div_yield": 0.020, "margin_est": 9000},
    "MYM=F": {"name": "Micro E-mini Dow Jones",   "mult": 0.50, "tick": 1.00, "tick_val": 0.50,  "div_yield": 0.020, "margin_est": 900},
}

# Map each symbol to its underlying index companion (for cross-correlation)
CROSS_SYMBOLS = {
    "NQ=F":  {"index": "NQ=F",  "cross": ["ES=F", "YM=F"]},
    "MNQ=F": {"index": "NQ=F",  "cross": ["ES=F", "YM=F"]},
    "ES=F":  {"index": "ES=F",  "cross": ["NQ=F", "YM=F"]},
    "MES=F": {"index": "ES=F",  "cross": ["NQ=F", "YM=F"]},
    "YM=F":  {"index": "YM=F",  "cross": ["NQ=F", "ES=F"]},
    "MYM=F": {"index": "YM=F",  "cross": ["NQ=F", "ES=F"]},
}

INDEX_SHORT = {"NQ=F": "NQ", "MNQ=F": "MNQ", "ES=F": "ES",
               "MES=F": "MES", "YM=F": "YM", "MYM=F": "MYM"}

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CME Equity Futures JD dashboard v1")
    p.add_argument("--symbol",   default="MNQ=F",
                   choices=list(CONTRACT_SPECS.keys()),
                   help="CME futures symbol (default: MNQ=F)")
    p.add_argument("--days",     type=int, default=500)
    p.add_argument("--show",     action="store_true")
    p.add_argument("--out",      default="eq_futures_jd_v1.png")
    p.add_argument("--synth",    action="store_true")
    p.add_argument("--no-carry", action="store_true",
                   help="Skip cost-of-carry adjustment")
    return p.parse_args()

# ─── EQ 1: Data — yfinance ────────────────────────────────────────────────────

def fetch_yf(symbol, days=500):
    """
    EQ 1: Fetch daily futures data via yfinance.
    Uses continuous front-month contract (e.g. NQ=F).
    Do NOT pass session= — yfinance ≥0.2.50 requires curl_cffi internally.
    SSL is fixed via CURL_CA_BUNDLE / SSL_CERT_FILE env vars above.
    """
    period = f"{max(days // 252 + 1, 2)}y"
    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbol}")
    # yfinance may return multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df.tail(days)

def fetch_vix(days=500):
    """Fetch VIX (^VIX) for fear gauge panel."""
    period = f"{max(days // 252 + 1, 2)}y"
    df = yf.download("^VIX", period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df["Close"].dropna().tail(days)

def fetch_tnx(days=500):
    """Fetch 10-Year Treasury yield (^TNX) for carry calculation."""
    period = f"{max(days // 252 + 1, 2)}y"
    df = yf.download("^TNX", period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df["Close"].dropna().tail(days)

def fetch_cross(cross_symbols, days=500):
    """Fetch cross-instrument data for correlation panel."""
    result = {}
    for sym in cross_symbols:
        try:
            period = f"{max(days // 252 + 1, 2)}y"
            df = yf.download(sym, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "date"
            s = df["Close"].dropna().tail(days)
            if len(s) > 30:
                result[sym] = s
        except Exception as e:
            print(f"  Cross {sym} failed: {e}")
    return result

# ─── EQ 2: Cost of carry ──────────────────────────────────────────────────────

def compute_carry(price_dates, tnx_series, div_yield, use_carry=True):
    """
    EQ 2: Daily cost of carry = (r_f − q) / 252
    r_f = 10Y treasury yield / 100
    q   = dividend yield (annualised, symbol-specific)
    Positive carry = long futures costs you (r_f > q is most regimes).
    Returns numpy array aligned to price_dates.
    """
    if not use_carry or tnx_series is None or len(tnx_series) == 0:
        return np.zeros(len(price_dates))
    idx = pd.DatetimeIndex(price_dates).normalize()
    rf  = tnx_series.reindex(idx, method="ffill").fillna(tnx_series.median()) / 100.
    daily_carry = (rf.values - div_yield) / 252
    return daily_carry

# ─── Synthetic data ───────────────────────────────────────────────────────────

def synthetic_equity(n=500, symbol="MNQ=F", seed=42):
    """
    Synthetic equity index futures with realistic parameters:
    lower vol (~20%/yr), moderate drift (~15%/yr), fewer/smaller jumps.
    """
    rng  = np.random.default_rng(seed)
    # Equity index params — much tamer than BTC
    TRUE = dict(mu=0.12, sigma=0.18, lam=6.0, mu_j=-0.025, sigma_j=0.035)
    dt   = 1/252
    r    = []
    for _ in range(n):
        nj = rng.poisson(TRUE["lam"]*dt)
        J  = np.sum(rng.normal(TRUE["mu_j"], TRUE["sigma_j"], nj)) if nj else 0.
        r.append((TRUE["mu"]-0.5*TRUE["sigma"]**2)*dt
                 + TRUE["sigma"]*np.sqrt(dt)*rng.normal() + J)
    spec = CONTRACT_SPECS.get(symbol, CONTRACT_SPECS["MNQ=F"])
    start_px = 21_000.  # ~NASDAQ level
    prices = start_px * np.exp(np.cumsum([0]+r))
    dates  = pd.date_range("2023-01-01", periods=n+1, freq="B")
    df = pd.DataFrame({
        "close":  prices,
        "volume": rng.exponential(50_000, n+1)
    }, index=dates)
    # Synthetic VIX: mean-reverting around 18, spikes on large neg returns
    vix = np.zeros(n)
    v = 18.0
    for i, ri in enumerate(r):
        v = 0.97*v + 0.03*18. + 2.*max(-ri*100, 0) + rng.normal(0, 0.5)
        v = max(v, 10.)
        vix[i] = v
    vix_s = pd.Series(vix, index=dates[1:])
    # Synthetic 10Y yield: ~4.3%
    tnx_vals = 4.3 + np.cumsum(rng.normal(0, 0.02, n))
    tnx_s    = pd.Series(np.clip(tnx_vals, 1., 8.), index=dates[1:])
    # Synthetic cross instruments (correlated)
    cross = {}
    for sym, px0, corr in [("ES=F", 5_500., 0.85), ("YM=F", 42_000., 0.80)]:
        if sym == symbol or sym.replace("=F","") in symbol:
            continue
        rc = [corr*ri + np.sqrt(1-corr**2)*rng.normal(0, TRUE["sigma"]*np.sqrt(dt))
              for ri in r]
        cross[sym] = pd.Series(px0 * np.exp(np.cumsum([0]+rc[:-1])), index=dates[1:])
    return df, vix_s, tnx_s, cross

# ─── Model dataclass ─────────────────────────────────────────────────────────

@dataclass
class JDParams:
    mu: float; sigma: float; lam: float; mu_j: float; sigma_j: float

    def effective_var(self):
        return self.sigma**2 + self.lam*(self.mu_j**2 + self.sigma_j**2)
    def effective_vol(self):
        return float(np.sqrt(self.effective_var()))
    def kelly_fraction(self, rf=0.):
        ev = self.effective_var()
        return 0. if ev < 1e-10 else (self.mu - rf)/ev
    def __str__(self):
        return textwrap.dedent(f"""
        JDParams
          mu       = {self.mu:+.4f}   (annualised drift, carry-adjusted)
          sigma    =  {self.sigma:.4f}   (diffusion vol)
          lambda   =  {self.lam:.3f}   (jumps/year)
          mu_j     = {self.mu_j:+.4f}   (mean log-jump)
          sigma_j  =  {self.sigma_j:.4f}   (jump size std)
          eff_vol  =  {self.effective_vol():.4f}
          kelly_f  =  {self.kelly_fraction():.4f}
        """).strip()

# ─── FIX 1: Lee-Mykland ──────────────────────────────────────────────────────

def lee_mykland(r: np.ndarray, significance: float = 0.001):
    n  = len(r)
    W  = max(10, min(22, n // 8))
    gs = float(np.std(r))
    bpv = (np.pi/2) * np.mean(np.abs(r[1:]) * np.abs(r[:-1]))
    rv  = float(np.mean(r**2))
    lv  = (pd.Series(r)
           .rolling(W, min_periods=max(5, W//4))
           .std().values)
    lv  = np.where(np.isnan(lv) | (lv < 1e-10), gs, lv)
    z   = r / lv
    cn  = np.sqrt(2.*np.log(n))
    sn  = cn - (np.log(np.pi) + np.log(np.log(n))) / (2.*cn)
    crit = sn - (1./cn)*np.log(-np.log(1. - significance))
    jm  = np.abs(z) > crit
    return {
        "jump_mask":  jm,
        "n_jumps":    int(jm.sum()),
        "jump_ratio": float(max(0., 1. - bpv/max(rv, 1e-12))),
        "z_scores":   z,
        "critical":   float(crit),
        "rv_annual":  rv*252,
        "bpv_annual": bpv*252,
        "window":     W,
    }

# ─── FIX 3: Regime detection ──────────────────────────────────────────────────

def detect_regimes(r: np.ndarray, n_iter: int = 60, seed: int = 0):
    rng  = np.random.default_rng(seed)
    n    = len(r)
    rv21 = (pd.Series(r).rolling(21, min_periods=5).std().values * np.sqrt(252))
    rv21 = np.where(np.isnan(rv21), np.nanmean(rv21[~np.isnan(rv21)]), rv21)
    obs  = rv21
    med  = np.median(obs)
    mu_s = np.array([obs[obs <= med].mean(), obs[obs > med].mean()])
    sg_s = np.array([obs[obs <= med].std() + 1e-6, obs[obs > med].std() + 1e-6])
    pi   = np.array([0.6, 0.4])
    A    = np.array([[0.95, 0.05],
                     [0.10, 0.90]])
    log_A = np.log(A + 1e-300)

    for _ in range(n_iter):
        log_em = np.column_stack([
            stats.norm.logpdf(obs, mu_s[s], sg_s[s]) for s in range(2)])
        log_alpha = np.empty((n, 2))
        log_alpha[0] = np.log(pi + 1e-300) + log_em[0]
        for t in range(1, n):
            for s in range(2):
                log_alpha[t, s] = (np.logaddexp(log_alpha[t-1, 0] + log_A[0, s],
                                                 log_alpha[t-1, 1] + log_A[1, s])
                                   + log_em[t, s])
        log_beta = np.zeros((n, 2))
        for t in range(n-2, -1, -1):
            for s in range(2):
                log_beta[t, s] = np.logaddexp(
                    log_A[s, 0] + log_em[t+1, 0] + log_beta[t+1, 0],
                    log_A[s, 1] + log_em[t+1, 1] + log_beta[t+1, 1])
        log_gamma = log_alpha + log_beta
        log_norm  = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])
        gamma     = np.exp(log_gamma - log_norm[:, np.newaxis])
        gamma     = np.clip(gamma, 1e-10, 1.)
        gamma    /= gamma.sum(axis=1, keepdims=True)
        log_xi = np.empty((n-1, 2, 2))
        for t in range(n-1):
            for s in range(2):
                for s2 in range(2):
                    log_xi[t, s, s2] = (log_alpha[t, s] + log_A[s, s2]
                                        + log_em[t+1, s2] + log_beta[t+1, s2])
            lnorm = log_xi[t].max()
            log_xi[t] -= lnorm + np.log(np.exp(log_xi[t] - log_xi[t].max()).sum())
        xi = np.exp(log_xi)
        xi = np.clip(xi, 1e-10, None)
        A  = xi.sum(axis=0)
        A /= A.sum(axis=1, keepdims=True)
        log_A = np.log(A + 1e-300)
        for s in range(2):
            w        = gamma[:, s]
            mu_s[s]  = (w * obs).sum() / w.sum()
            sg_s[s]  = np.sqrt((w * (obs - mu_s[s])**2).sum() / w.sum()) + 1e-6
        pi = gamma[0] / gamma[0].sum()

    if mu_s[0] > mu_s[1]:
        gamma  = gamma[:, ::-1]
        mu_s   = mu_s[::-1]
        sg_s   = sg_s[::-1]

    state_seq = np.argmax(gamma, axis=1)
    params = {}
    for s, label in enumerate(["Bull", "Bear"]):
        mask = state_seq == s
        rs   = r[mask] if mask.sum() > 5 else r
        params[label] = {
            "mu_ann":  float(np.mean(rs) * 252),
            "vol_ann": float(np.std(rs) * np.sqrt(252)),
            "n_days":  int(mask.sum()),
            "pct":     float(mask.mean() * 100),
        }
    return gamma, state_seq, params

# ─── Method of Moments ───────────────────────────────────────────────────────

def method_of_moments(r: np.ndarray, dt: float = 1/252) -> JDParams:
    m1   = float(np.mean(r))
    var  = float(np.var(r))
    sk   = float(stats.skew(r))
    ku   = float(stats.kurtosis(r))
    p95  = float(np.percentile(np.abs(r), 95))
    p50  = float(np.percentile(np.abs(r), 50))
    jp   = max(p95 - p50, 1e-5)
    lam     = float(np.clip((ku*var)/(3.*jp**2*dt+1e-10)*dt, 0.5, 52.))
    mu_j    = float(np.clip((sk*var**1.5)/(3.*lam*dt*jp**2+1e-10), -0.5, 0.5))
    sigma_j = float(np.clip(np.sqrt(max(ku*var**2/(3.*lam+1e-10)/dt, 1e-6)), 0.01, 1.))
    jvc     = lam*dt*(mu_j**2 + sigma_j**2)
    sigma   = float(np.clip(np.sqrt(max(var-jvc, var*0.05)/dt), 0.05, 5.))
    mu      = float(np.clip(m1/dt - lam*mu_j + 0.5*sigma**2, -5., 15.))
    return JDParams(mu=mu, sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)

# ─── FIX 2: MLE ──────────────────────────────────────────────────────────────

def _merton_nll(pv, r, dt, k_max=20):
    mu, log_sig, log_lam, mu_j, log_sj = pv
    sig = np.exp(log_sig); lam = np.exp(log_lam); sj = np.exp(log_sj)
    ld  = lam * dt
    penalty = max(0., log_lam - np.log(20.))**2 * 50.
    lf = np.zeros(k_max+1)
    for k in range(1, k_max+1):
        lf[k] = lf[k-1] + np.log(k)
    n  = len(r)
    lc = np.empty((k_max+1, n))
    for k in range(k_max+1):
        lw   = k*np.log(ld+1e-300) - ld - lf[k]
        mk   = (mu - 0.5*sig**2)*dt + k*mu_j
        vk   = max(sig**2*dt + k*sj**2, 1e-12)
        lc[k] = lw - 0.5*(np.log(2.*np.pi*vk)) - 0.5*(r - mk)**2/vk
    mc  = lc.max(axis=0)
    lml = mc + np.log(np.sum(np.exp(lc - mc[np.newaxis,:]), axis=0))
    nll = -float(np.sum(lml))
    return (nll + penalty) if np.isfinite(nll) else 1e12

BOUNDS = [
    (-5., 15.),
    (np.log(0.01), np.log(3.)),        # equity vol rarely > 200%
    (np.log(0.05), np.log(52.)),
    (-0.5, 0.5),                        # equity jumps smaller than BTC
    (np.log(0.005), np.log(1.)),        # equity jump size smaller
]

def mle_calibrate(r, dt=1/252, init=None, n_restarts=4):
    if init is None:
        init = method_of_moments(r, dt)
    x0 = np.array([
        init.mu,
        np.log(max(init.sigma,   0.05)),
        np.log(max(init.lam,     0.5)),
        init.mu_j,
        np.log(max(init.sigma_j, 0.01)),
    ])
    rng = np.random.default_rng(42)
    best_nll, best_x = np.inf, None
    for i in range(n_restarts):
        xs = x0.copy() if i == 0 else np.clip(
            x0 + rng.normal(0., 0.30, len(x0)),
            [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = optimize.minimize(
                    _merton_nll, xs, args=(r, dt),
                    method="L-BFGS-B", bounds=BOUNDS,
                    options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9})
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll, best_x = res.fun, res.x.copy()
        except Exception:
            continue
    if best_x is None:
        return init
    return JDParams(
        mu=float(best_x[0]), sigma=float(np.exp(best_x[1])),
        lam=float(np.exp(best_x[2])), mu_j=float(best_x[3]),
        sigma_j=float(np.exp(best_x[4])))

# ─── Regime signal ───────────────────────────────────────────────────────────

def regime_signal(r, gamma, dt=1/252, n_restarts=4):
    state_seq = np.argmax(gamma, axis=1)
    results   = {}
    for s, label in enumerate(["Bull", "Bear"]):
        mask = state_seq == s
        rs   = r[mask]
        if len(rs) < 30:
            rs = r
        init = method_of_moments(rs, dt)
        p    = mle_calibrate(rs, dt, init, n_restarts)
        results[label] = p
    p_bull  = float(gamma[-1, 0])
    p_bear  = float(gamma[-1, 1])
    k_bull  = results["Bull"].kelly_fraction()
    k_bear  = results["Bear"].kelly_fraction()
    k_blend = p_bull * k_bull + p_bear * k_bear
    return results, p_bull, p_bear, k_blend

# ─── Rolling calibration ─────────────────────────────────────────────────────

def rolling_calibrate(r, dt=1/252, window=90, step=10):
    records = []
    for end in range(window, len(r)+1, step):
        sl = r[end-window:end]
        try:
            init = method_of_moments(sl, dt)
            p    = mle_calibrate(sl, dt, init, n_restarts=2)
            records.append({
                "bar":     end-1,
                "mu":      round(p.mu, 4),
                "sigma":   round(p.sigma, 4),
                "lam":     round(p.lam, 3),
                "eff_vol": round(p.effective_vol(), 4),
                "kelly":   round(np.clip(p.kelly_fraction(), -15, 15), 4),
            })
        except Exception:
            continue
    return pd.DataFrame(records).set_index("bar")

# ─── JD density ──────────────────────────────────────────────────────────────

def jd_density(x, p, dt=1/252, k_max=20):
    ld = p.lam*dt
    lf = np.zeros(k_max+1)
    for k in range(1, k_max+1): lf[k] = lf[k-1]+np.log(k)
    d  = np.zeros_like(x, dtype=float)
    for k in range(k_max+1):
        w  = np.exp(k*np.log(ld+1e-300) - ld - lf[k])
        mk = (p.mu - 0.5*p.sigma**2)*dt + k*p.mu_j
        vk = max(p.sigma**2*dt + k*p.sigma_j**2, 1e-12)
        d += w * stats.norm.pdf(x, mk, np.sqrt(vk))
    return d

# ─── EQ 3: VIX regime levels ──────────────────────────────────────────────────

VIX_LEVELS = [
    (12,  "COMPLACENCY",  "#00e676", 0.05),
    (20,  "CALM",         "#29b6f6", 0.05),
    (30,  "ELEVATED",     "#ffb300", 0.05),
    (40,  "FEAR",         "#ff3d5a", 0.05),
    (999, "PANIC",        "#ff0000", 0.08),
]

def vix_label(vix_val):
    for threshold, label, col, _ in VIX_LEVELS:
        if vix_val < threshold:
            return label, col
    return "PANIC", "#ff0000"

# ─── EQ 4: Cross correlation ──────────────────────────────────────────────────

def rolling_cross_corr(r_main, cross_data, dates, window=21):
    """Rolling 21d correlation between main instrument and cross instruments."""
    result = {}
    main_s = pd.Series(r_main, index=dates)
    for sym, px_series in cross_data.items():
        r_cross = np.diff(np.log(px_series.values))
        cross_dates = px_series.index[1:]
        cross_s = pd.Series(r_cross, index=cross_dates)
        # Align
        aligned = pd.DataFrame({"main": main_s, "cross": cross_s}).dropna()
        if len(aligned) < window + 5:
            continue
        corr = aligned["main"].rolling(window).corr(aligned["cross"])
        result[sym] = corr
    return result

# ─── Colours ─────────────────────────────────────────────────────────────────

DARK  = "#060b18"; CARD = "#0d1526"; DIM = "#18273f"
GREEN = "#00e676"; RED  = "#ff3d5a"; AMBER = "#ffb300"
BLUE  = "#29b6f6"; TEXT = "#c8d8f0"; MUTED = "#3d5275"
BULL_C = "#00e676"; BEAR_C = "#ff3d5a"
CARRY_C = "#ff9800"   # positive carry = cost for long

CROSS_COLOURS = {"NQ=F": "#29b6f6", "ES=F": "#00e676",
                 "YM=F": "#ffb300", "MNQ=F": "#29b6f6",
                 "MES=F": "#00e676", "MYM=F": "#ffb300"}

def _ax(ax, title=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(DIM)
    ax.grid(color=DIM, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=MUTED, fontsize=8, loc="left",
                     fontfamily="monospace", pad=6)

# ─── Dashboard ───────────────────────────────────────────────────────────────

def build_dashboard(df, r_raw, r_adj, carry_daily,
                    params_full, lm_res, roll,
                    gamma, state_seq, regime_params,
                    p_bull, p_bear, k_blend,
                    reg_results, vix_series, cross_corrs,
                    use_carry, symbol, spec,
                    out_path, show):

    prices   = df["close"].values
    dates    = df.index
    td       = dates[1:]
    jm       = lm_res["jump_mask"]

    K_full   = params_full.kelly_fraction()
    K_blend  = k_blend
    pos      = float(np.clip(K_blend * 0.25, 0., 0.25))
    signal   = "LONG" if K_blend > 0.4 else "SHORT" if K_blend < -0.3 else "FLAT"
    sig_c    = GREEN if signal == "LONG" else RED if signal == "SHORT" else AMBER

    # Contract-level position info
    last_px    = float(prices[-1])
    notional   = last_px * spec["mult"]
    tick_val   = spec["tick_val"]
    quarter_k_lots = max(1, int(pos / (tick_val / notional * 100)))  # rough

    matplotlib.rcParams.update({
        "font.family": "monospace", "text.color": TEXT,
        "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "figure.facecolor": DARK,
    })

    fig = plt.figure(figsize=(20, 13), facecolor=DARK)
    pct_chg   = (prices[-1]/prices[-2]-1)*100
    last_date = dates[-1].strftime("%Y-%m-%d")
    chg_c     = GREEN if pct_chg >= 0 else RED

    # VIX current level
    last_vix = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str  = f"VIX={last_vix:.1f}" if last_vix else "VIX=N/A"
    vlabel, vcol = vix_label(last_vix) if last_vix else ("N/A", MUTED)

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.01, 0.965, f"{INDEX_SHORT.get(symbol, symbol)}", fontsize=22,
             color=GREEN, fontweight="bold", fontfamily="monospace")
    fig.text(0.09, 0.968, f"{last_px:,.2f}", fontsize=20, color=TEXT,
             fontfamily="monospace")
    fig.text(0.22, 0.970, f"{'+' if pct_chg>=0 else ''}{pct_chg:.2f}% 1D",
             fontsize=13, color=chg_c, fontfamily="monospace")
    fig.text(0.34, 0.970, vix_str, fontsize=13, color=vcol,
             fontfamily="monospace")
    fig.text(0.46, 0.970, vlabel, fontsize=11, color=vcol,
             fontfamily="monospace")
    adj_note = "CARRY-ADJUSTED returns" if use_carry else "RAW returns"
    fig.text(0.01, 0.946,
             f"MERTON JD v1  |  CME FUTURES: {spec['name']}  |  "
             f"{len(r_adj)} obs  |  {last_date}  |  {adj_note}  |  "
             f"mult=${spec['mult']}  tick=${spec['tick_val']:.2f}  notional≈${notional:,.0f}",
             fontsize=8.5, color=MUTED, fontfamily="monospace")

    # ── Signal strip ──────────────────────────────────────────────────────────
    bull_p = reg_results["Bull"]
    bear_p = reg_results["Bear"]
    regime_now = "BULL" if p_bull > p_bear else "BEAR"
    regime_c   = BULL_C if regime_now == "BULL" else BEAR_C
    ann_carry  = float(np.mean(carry_daily)) * 252 * 100 if use_carry else 0.
    carry_note = f"ann.carry={ann_carry:+.2f}%/yr" if use_carry else "carry OFF"

    sig_line = (
        f"SIGNAL: {signal}   Kelly(blend)={K_blend:+.3f}   Kelly(full)={K_full:+.3f}   "
        f"¼-Kelly={pos*100:.1f}%   regime={regime_now}  p_bull={p_bull:.2f}  p_bear={p_bear:.2f}  "
        f"μ={params_full.mu:+.3f}  σ={params_full.sigma:.3f}  λ={params_full.lam:.2f}  "
        f"effVol={params_full.effective_vol():.3f}  jumps={lm_res['n_jumps']}  {carry_note}"
    )
    fig.text(0.01, 0.924, sig_line, fontsize=8.8, color=sig_c,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD,
                       edgecolor=sig_c, linewidth=1.3, alpha=0.9))

    # ── Regime comparison strip ────────────────────────────────────────────────
    rp = regime_params
    bull_line = (f"[BULL {rp['Bull']['pct']:.0f}%]  "
                 f"μ={rp['Bull']['mu_ann']:+.3f}  vol={rp['Bull']['vol_ann']:.3f}  "
                 f"Kelly={bull_p.kelly_fraction():+.3f}  "
                 f"λ={bull_p.lam:.2f}  μⱼ={bull_p.mu_j:+.3f}")
    bear_line = (f"[BEAR {rp['Bear']['pct']:.0f}%]  "
                 f"μ={rp['Bear']['mu_ann']:+.3f}  vol={rp['Bear']['vol_ann']:.3f}  "
                 f"Kelly={bear_p.kelly_fraction():+.3f}  "
                 f"λ={bear_p.lam:.2f}  μⱼ={bear_p.mu_j:+.3f}")
    fig.text(0.01, 0.905, bull_line, fontsize=8, color=BULL_C, fontfamily="monospace")
    fig.text(0.45, 0.905, bear_line, fontsize=8, color=BEAR_C, fontfamily="monospace")

    # ── Grid ─────────────────────────────────────────────────────────────────
    gs2 = gridspec.GridSpec(3, 3, figure=fig,
                            top=0.895, bottom=0.05, left=0.05, right=0.97,
                            hspace=0.55, wspace=0.32)
    axes = [[fig.add_subplot(gs2[r, c]) for c in range(3)] for r in range(3)]

    fmtp = plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%")

    # ── P1: Price + jumps + regime ────────────────────────────────────────────
    a = axes[0][0]
    _ax(a, f"{INDEX_SHORT.get(symbol,symbol)} FUTURES PRICE  +  REGIME  +  JUMP EVENTS ▼")
    in_bear = False; bear_start = None
    for i, s in enumerate(state_seq):
        if s == 1 and not in_bear:
            in_bear = True; bear_start = td[i]
        elif s == 0 and in_bear:
            a.axvspan(bear_start, td[i], color=RED, alpha=0.08)
            in_bear = False
    if in_bear:
        a.axvspan(bear_start, td[-1], color=RED, alpha=0.08)
    a.plot(dates, prices, color=BLUE, linewidth=1.2, alpha=0.9)
    a.fill_between(dates, prices, alpha=0.06, color=BLUE)
    jdates = td[jm]; jpx = prices[1:][jm]
    a.scatter(jdates, jpx, color=RED, s=30, marker="v", zorder=5,
              label=f"{lm_res['n_jumps']} jumps (LM)")
    a.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    a.tick_params(axis="x", rotation=20, labelsize=7)
    a.legend(fontsize=8, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)

    # ── P2: Return distribution — raw AND carry-adj overlaid ──────────────────
    a = axes[0][1]
    _ax(a, "RETURN DIST  —  raw vs carry-adj  vs  JD fit")
    bins_common = np.linspace(
        min(r_raw.min(), r_adj.min()),
        max(r_raw.max(), r_adj.max()), 60)
    a.hist(r_raw, bins=bins_common, density=True, color=MUTED, alpha=0.3,
           label="Raw returns")
    cnt, bins2, patches = a.hist(r_adj, bins=bins_common, density=True,
                                  color=BLUE, alpha=0.45, label="Carry-adj")
    for patch, left in zip(patches, bins2[:-1]):
        if left < -0.02:
            patch.set_facecolor(RED); patch.set_alpha(0.55)
    xg = np.linspace(bins2[0], bins2[-1], 400)
    a.plot(xg, jd_density(xg, params_full), color=GREEN, linewidth=2.2, label="JD full")
    a.plot(xg, jd_density(xg, bull_p),  color=BULL_C, linewidth=1.2,
           linestyle=":", alpha=0.8, label="JD bull")
    a.plot(xg, jd_density(xg, bear_p),  color=BEAR_C, linewidth=1.2,
           linestyle=":", alpha=0.8, label="JD bear")
    a.plot(xg, stats.norm.pdf(xg, np.mean(r_adj), np.std(r_adj)),
           color=AMBER, linewidth=1., linestyle="--", alpha=0.6, label="Normal")
    a.set_xlabel("Log return", fontsize=8, color=MUTED)
    a.legend(fontsize=7, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)

    # ── P3: Regime posterior ──────────────────────────────────────────────────
    a = axes[0][2]
    _ax(a, "REGIME POSTERIOR  p(Bull) / p(Bear)  [HMM]")
    a.fill_between(td, gamma[:,0], color=BULL_C, alpha=0.4, label="p(Bull)")
    a.fill_between(td, gamma[:,1], color=BEAR_C, alpha=0.3, label="p(Bear)")
    a.plot(td, gamma[:,0], color=BULL_C, linewidth=1.)
    a.plot(td, gamma[:,1], color=BEAR_C, linewidth=1.)
    a.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    a.set_ylim(0, 1)
    a.yaxis.set_major_formatter(fmtp)
    a.tick_params(axis="x", rotation=20, labelsize=7)
    a.legend(fontsize=8, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)
    a.text(td[-1], float(gamma[-1,0])+0.03,
           f" {p_bull:.0%}", color=BULL_C, fontsize=8, va="bottom",
           fontfamily="monospace")

    # ── P4: Lee-Mykland Z-scores ──────────────────────────────────────────────
    a = axes[1][0]
    _ax(a, f"LM Z-SCORES (carry-adj, sig=0.001  crit=±{lm_res['critical']:.2f}  win={lm_res['window']}d)")
    z  = lm_res["z_scores"]
    bc = [RED if j else (BLUE if zi >= 0 else MUTED) for zi,j in zip(z,jm)]
    a.bar(td, z, color=bc, alpha=0.75, width=1.)
    a.axhline( lm_res["critical"], color=RED, linewidth=1., linestyle="--", alpha=0.7)
    a.axhline(-lm_res["critical"], color=RED, linewidth=1., linestyle="--", alpha=0.7)
    a.axhline(0, color=MUTED, linewidth=0.7)
    a.set_ylabel("z-score", fontsize=8, color=MUTED)
    a.tick_params(axis="x", rotation=20, labelsize=7)

    # ── P5: Rolling effective vol ─────────────────────────────────────────────
    a = axes[1][1]
    _ax(a, "ROLLING EFF. VOL  vs  DIFFUSION σ  (90d, carry-adj)")
    rb = roll.index
    a.plot(rb, roll["eff_vol"], color=BLUE,  linewidth=1.8, label="Eff. vol")
    a.plot(rb, roll["sigma"],   color=AMBER, linewidth=1.2, linestyle="--", label="Diffusion σ")
    a.fill_between(rb, roll["sigma"], roll["eff_vol"], alpha=0.1, color=BLUE, label="Jump var")
    a.yaxis.set_major_formatter(fmtp)
    # Overlay VIX on twin axis if available
    if vix_series is not None and len(vix_series) > 0:
        ax2 = a.twinx()
        vix_aligned = vix_series.reindex(
            pd.DatetimeIndex([dates[r] for r in rb]), method="nearest")
        ax2.plot(rb, vix_aligned.values, color=RED, linewidth=0.9,
                 alpha=0.5, linestyle="-.", label="VIX")
        ax2.set_ylabel("VIX", fontsize=7, color=RED)
        ax2.tick_params(colors=MUTED, labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(DIM)
        ax2.axhline(20, color=RED, linewidth=0.5, linestyle=":", alpha=0.4)
    a.set_xlabel("Bar", fontsize=8, color=MUTED)
    a.legend(fontsize=8, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)

    # ── P6: Rolling Kelly ─────────────────────────────────────────────────────
    a = axes[1][2]
    _ax(a, "ROLLING KELLY FRACTION  (90d, carry-adj)")
    kv = roll["kelly"].values
    a.bar(rb, kv, color=[GREEN if k>=0 else RED for k in kv], alpha=0.75, width=4.)
    a.axhline(0,    color=MUTED, linewidth=0.8)
    a.axhline( 0.4, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.5, label="Long thr.")
    a.axhline(-0.3, color=RED,   linewidth=0.8, linestyle=":", alpha=0.5, label="Short thr.")
    a.set_xlabel("Bar", fontsize=8, color=MUTED)
    a.legend(fontsize=8, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)

    # ── P7: VIX history + regime zones  (EQ 3) ────────────────────────────────
    a = axes[2][0]
    _ax(a, "VIX  —  COMPLACENCY → PANIC  +  HMM REGIME OVERLAY")
    if vix_series is not None and len(vix_series) > 0:
        vix_dates = vix_series.index
        vix_vals  = vix_series.values
        # Shade VIX zones
        prev_thresh = 0
        zone_colours = [(12, "#00e676", 0.04), (20, "#29b6f6", 0.04),
                        (30, "#ffb300", 0.05), (40, "#ff3d5a", 0.06),
                        (80, "#ff0000", 0.08)]
        for thresh, col, alp in zone_colours:
            a.axhspan(prev_thresh, thresh, color=col, alpha=alp)
            prev_thresh = thresh
        # Annotate zone lines
        for thresh, label, col, _ in VIX_LEVELS[1:]:
            if thresh < 80:
                a.axhline(thresh, color=col, linewidth=0.7, linestyle="--", alpha=0.5)
                a.text(vix_dates[2], thresh+0.5, label, color=col, fontsize=7,
                       fontfamily="monospace")
        # Bear regime overlay on VIX
        in_bear = False; bear_start_v = None
        for i, s in enumerate(state_seq):
            if i >= len(vix_dates): break
            if s == 1 and not in_bear:
                in_bear = True; bear_start_v = vix_dates[min(i, len(vix_dates)-1)]
            elif s == 0 and in_bear:
                a.axvspan(bear_start_v, vix_dates[min(i, len(vix_dates)-1)],
                          color=RED, alpha=0.08)
                in_bear = False
        if in_bear and bear_start_v is not None:
            a.axvspan(bear_start_v, vix_dates[-1], color=RED, alpha=0.08)
        a.fill_between(vix_dates, vix_vals, alpha=0.25, color=AMBER)
        a.plot(vix_dates, vix_vals, color=AMBER, linewidth=1.4)
        # Rolling 21d VIX percentile
        vix_pct = pd.Series(vix_vals, index=vix_dates).rank(pct=True).values * 100
        ax2 = a.twinx()
        ax2.plot(vix_dates, vix_pct, color=BLUE, linewidth=0.9,
                 alpha=0.6, linestyle="--", label="VIX pct")
        ax2.set_ylabel("VIX percentile %", fontsize=7, color=BLUE)
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors=MUTED, labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(DIM)
        if last_vix:
            a.axhline(last_vix, color=vcol, linewidth=1., linestyle=":")
            a.text(vix_dates[-1], last_vix+0.5, f"  {last_vix:.1f}",
                   color=vcol, fontsize=9, fontfamily="monospace")
        a.tick_params(axis="x", rotation=20, labelsize=7)
        a.set_ylabel("VIX", fontsize=8, color=AMBER)
    else:
        a.text(0.5, 0.5, "VIX data unavailable", ha="center", va="center",
               color=MUTED, fontsize=9, fontfamily="monospace",
               transform=a.transAxes)

    # ── P8: Cross-instrument correlation  (EQ 4) ──────────────────────────────
    a = axes[2][1]
    _ax(a, "CROSS-INDEX CORRELATION  (rolling 21d)  NQ / ES / YM")
    if cross_corrs:
        for sym, corr_s in cross_corrs.items():
            col = CROSS_COLOURS.get(sym, BLUE)
            short = INDEX_SHORT.get(sym, sym)
            a.plot(corr_s.index, corr_s.values, color=col,
                   linewidth=1.4, label=f"vs {short}", alpha=0.85)
        a.axhline(1.0,  color=MUTED, linewidth=0.6, linestyle="--")
        a.axhline(0.7,  color=AMBER, linewidth=0.7, linestyle=":", alpha=0.5,
                  label="Corr=0.7 (divergence alert)")
        a.axhline(0.0,  color=MUTED, linewidth=0.6)
        a.set_ylim(-0.2, 1.05)
        a.set_ylabel("Pearson r", fontsize=8, color=MUTED)
        a.tick_params(axis="x", rotation=20, labelsize=7)
        a.legend(fontsize=8, facecolor=CARD, edgecolor=DIM, labelcolor=TEXT)
        # Note: correlation breakdown = de-correlation = stress signal
        a.text(0.01, 0.06,
               "de-corr below 0.7 = inter-market stress",
               transform=a.transAxes, fontsize=7, color=AMBER,
               fontfamily="monospace")
    else:
        a.text(0.5, 0.5,
               "Cross-instrument data unavailable",
               ha="center", va="center", color=MUTED, fontsize=9,
               fontfamily="monospace", transform=a.transAxes)

    # ── P9: Position sizing with contract specs ────────────────────────────────
    a = axes[2][2]
    _ax(a, "POSITION SIZING  +  CONTRACT SPECS")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.axis("off")

    def pbox(ax, x, y, w, h, label, val, col, note=""):
        ax.add_patch(plt.Rectangle((x,y), w, h, facecolor=CARD,
                                    edgecolor=col, linewidth=1.5))
        ax.text(x+w/2, y+h*0.72, label, ha="center", va="center",
                fontsize=8, color=MUTED, fontfamily="monospace")
        ax.text(x+w/2, y+h*0.38, val, ha="center", va="center",
                fontsize=14, color=col, fontweight="bold", fontfamily="monospace")
        if note:
            ax.text(x+w/2, y+h*0.12, note, ha="center", va="center",
                    fontsize=7, color=MUTED, fontfamily="monospace")

    carry_str = f"ann carry≈{ann_carry:+.2f}%/yr" if use_carry else "carry adj OFF"
    pbox(a, 0.02, 0.55, 0.45, 0.38, "SIGNAL",      signal,           sig_c)
    pbox(a, 0.53, 0.55, 0.45, 0.38, "¼-KELLY SIZE",f"{pos*100:.1f}%", sig_c, "max 25%")
    pbox(a, 0.02, 0.10, 0.45, 0.38, "REGIME NOW",  regime_now,        regime_c,
         f"p={max(p_bull,p_bear):.0%}")
    pbox(a, 0.53, 0.10, 0.45, 0.38, "BLEND KELLY", f"{K_blend:+.3f}", sig_c,
         carry_str)

    # Contract spec box
    spec_text = (
        f"Contract: {spec['name']}\n"
        f"Multiplier: ${spec['mult']}/pt\n"
        f"Tick: {spec['tick']} pt = ${spec['tick_val']:.2f}\n"
        f"Notional @ {last_px:,.0f}: ${notional:,.0f}\n"
        f"Margin est.: ~${spec['margin_est']:,}\n"
        f"Div yield est.: {spec['div_yield']*100:.1f}%/yr"
    )
    a.text(0.5, 0.52, spec_text, ha="center", va="top",
           fontsize=7.5, color=MUTED, fontfamily="monospace",
           transform=a.transAxes, linespacing=1.6)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.018,
             f"Merton (1976) JD  |  CME {spec['name']}  |  "
             f"yfinance data  |  Carry-adj returns  |  LM sig=0.001  |  "
             f"2-state HMM  |  ¼-Kelly max 25%  |  Not financial advice",
             ha="center", fontsize=8, color=MUTED, fontfamily="monospace")

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    print(f"\n  ✓ Saved → {out_path}")
    if show:
        plt.show()
    plt.close()

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    SEP     = "─" * 64
    symbol  = args.symbol.upper()
    spec    = CONTRACT_SPECS[symbol]
    use_adj = not args.no_carry
    dt      = 1/252

    print(SEP)
    print(f"  {spec['name']}  ({symbol})  —  JD Dashboard  Futures v1")
    print(f"  Multiplier: ${spec['mult']}/pt  Tick: ${spec['tick_val']:.2f}")
    print(SEP)

    # ── EQ 1: Data ────────────────────────────────────────────────────────────
    print("\n[DATA]")
    vix_series  = None
    tnx_series  = None
    cross_data  = {}

    if args.synth:
        print(f"  Using synthetic {symbol} data")
        df, vix_series, tnx_series, cross_data = synthetic_equity(
            n=args.days, symbol=symbol)
    else:
        print(f"  Fetching {symbol} from yfinance…")
        df = fetch_yf(symbol, args.days)
        print(f"  ✓ {symbol}: {len(df)} bars  "
              f"range {float(df['close'].min()):,.1f}–{float(df['close'].max()):,.1f}")

        print("  Fetching VIX (^VIX)…")
        try:
            vix_series = fetch_vix(args.days)
            print(f"  ✓ VIX: {len(vix_series)} bars  last={float(vix_series.iloc[-1]):.2f}")
        except Exception as e:
            print(f"  VIX failed: {e}")

        if use_adj:
            print("  Fetching 10Y yield (^TNX) for carry…")
            try:
                tnx_series = fetch_tnx(args.days)
                print(f"  ✓ TNX: {len(tnx_series)} bars  "
                      f"last={float(tnx_series.iloc[-1]):.2f}%")
            except Exception as e:
                print(f"  TNX failed: {e}  — using div yield only (r_f=0)")
                use_adj = True           # still adjust, but rf=0 → carry = -q

        print("  Fetching cross-instrument data…")
        cross_syms = CROSS_SYMBOLS.get(symbol, {}).get("cross", [])
        cross_data = fetch_cross(cross_syms, args.days)
        print(f"  ✓ Cross: {list(cross_data.keys())}")

    prices = df["close"].values
    r_raw  = np.diff(np.log(prices))
    print(f"\n  Bars  : {len(r_raw)}")
    print(f"  Price : {prices.min():,.2f} – {prices.max():,.2f}")
    print(f"  μ_raw : {np.mean(r_raw)*252:+.3f}/yr")
    print(f"  σ_raw : {np.std(r_raw)*np.sqrt(252):.3f}/yr")

    # ── EQ 2: Carry adjustment ────────────────────────────────────────────────
    return_dates = df.index[1:]
    carry_daily  = compute_carry(return_dates, tnx_series,
                                 spec["div_yield"], use_carry=use_adj)
    r_adj        = r_raw - carry_daily if use_adj else r_raw.copy()
    r_model      = r_adj

    if use_adj:
        ann_carry = float(np.mean(carry_daily)) * 252 * 100
        print(f"\n  Carry adj ON  |  ann carry shift: {ann_carry:+.3f}%/yr")
        print(f"  μ_adj : {np.mean(r_adj)*252:+.3f}/yr")

    # ── FIX 1: Lee-Mykland ────────────────────────────────────────────────────
    print(f"\n[1/4] Lee-Mykland  (sig=0.001, carry-adj returns)")
    lm = lee_mykland(r_model, significance=0.001)
    print(f"  Jumps     : {lm['n_jumps']}  ({lm['n_jumps']/len(r_model)*100:.1f}%)")
    print(f"  Jump var% : {lm['jump_ratio']*100:.1f}%")
    print(f"  Critical z: {lm['critical']:.3f}  window={lm['window']}d")

    # ── FIX 3: Regime detection ───────────────────────────────────────────────
    print(f"\n[2/4] HMM regime detection (2 states, 60 EM iterations)")
    gamma, state_seq, regime_params = detect_regimes(r_model)
    print(f"  Bull days : {regime_params['Bull']['n_days']}  "
          f"({regime_params['Bull']['pct']:.1f}%)  "
          f"μ={regime_params['Bull']['mu_ann']:+.3f}  "
          f"vol={regime_params['Bull']['vol_ann']:.3f}")
    print(f"  Bear days : {regime_params['Bear']['n_days']}  "
          f"({regime_params['Bear']['pct']:.1f}%)  "
          f"μ={regime_params['Bear']['mu_ann']:+.3f}  "
          f"vol={regime_params['Bear']['vol_ann']:.3f}")
    p_bull = float(gamma[-1, 0])
    p_bear = float(gamma[-1, 1])
    print(f"  Current   : p_bull={p_bull:.2f}  p_bear={p_bear:.2f}  "
          f"→ {'BULL' if p_bull>p_bear else 'BEAR'} regime")

    # ── Full MLE ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Full-sample MLE (carry-adj, 4 restarts, λ-bound=52)")
    init        = method_of_moments(r_model, dt)
    params_full = mle_calibrate(r_model, dt, init, n_restarts=4)
    print(f"\n{params_full}\n")

    # ── Regime calibration ────────────────────────────────────────────────────
    print(f"[4/4] Regime-separated calibration + blended Kelly")
    reg_results, p_bull, p_bear, k_blend = regime_signal(r_model, gamma, dt)
    print(f"  Bull JD   : {reg_results['Bull']}")
    print(f"  Bear JD   : {reg_results['Bear']}")
    print(f"  Kelly blend: {k_blend:+.4f}  "
          f"= {p_bull:.2f}×{reg_results['Bull'].kelly_fraction():+.3f}"
          f" + {p_bear:.2f}×{reg_results['Bear'].kelly_fraction():+.3f}")

    # ── Rolling calibration ───────────────────────────────────────────────────
    print(f"\n[5/5] Rolling calibration (window=90, step=10)…")
    roll = rolling_calibrate(r_model, dt)
    print(f"  Windows: {len(roll)}  |  "
          f"Kelly {roll['kelly'].min():.3f}→{roll['kelly'].max():.3f}  |  "
          f"λ {roll['lam'].min():.2f}→{roll['lam'].max():.2f}")

    # ── Cross correlations ────────────────────────────────────────────────────
    cross_corrs = rolling_cross_corr(r_model, cross_data, return_dates, window=21)

    # ── Final signal ──────────────────────────────────────────────────────────
    pos    = float(np.clip(k_blend * 0.25, 0., 0.25))
    signal = "LONG" if k_blend > 0.4 else "SHORT" if k_blend < -0.3 else "FLAT"
    notional = float(prices[-1]) * spec["mult"]
    print(f"\n{SEP}")
    print(f"  SIGNAL        : {signal}")
    print(f"  Kelly (blend) : {k_blend:+.4f}")
    print(f"  Kelly (full)  : {params_full.kelly_fraction():+.4f}")
    print(f"  ¼-Kelly size  : {pos*100:.1f}%  (max 25%)")
    print(f"  Regime now    : {'BULL' if p_bull>p_bear else 'BEAR'}  "
          f"p_bull={p_bull:.2f}  p_bear={p_bear:.2f}")
    if use_adj:
        ann_carry = float(np.mean(carry_daily)) * 252 * 100
        print(f"  Ann. carry    : {ann_carry:+.3f}%/yr")
    print(f"\n  Contract      : {spec['name']}")
    print(f"  Notional/lot  : ${notional:,.0f}  "
          f"(${spec['mult']}/pt × {float(prices[-1]):,.1f})")
    print(f"  Tick value    : ${spec['tick_val']:.2f}")
    print(f"  Margin est.   : ${spec['margin_est']:,}")
    if vix_series is not None and len(vix_series) > 0:
        last_vix = float(vix_series.iloc[-1])
        vlabel, _ = vix_label(last_vix)
        print(f"  VIX           : {last_vix:.1f}  [{vlabel}]")
    if cross_corrs:
        for sym, corr_s in cross_corrs.items():
            last_c = float(corr_s.dropna().iloc[-1]) if len(corr_s.dropna()) > 0 else np.nan
            print(f"  Corr vs {INDEX_SHORT.get(sym,sym):4s}  : {last_c:.3f}")
    print(SEP)

    print("\n[CHART] Building dashboard…")
    build_dashboard(
        df=df, r_raw=r_raw, r_adj=r_adj, carry_daily=carry_daily,
        params_full=params_full, lm_res=lm, roll=roll,
        gamma=gamma, state_seq=state_seq, regime_params=regime_params,
        p_bull=p_bull, p_bear=p_bear, k_blend=k_blend,
        reg_results=reg_results,
        vix_series=vix_series, cross_corrs=cross_corrs,
        use_carry=use_adj, symbol=symbol, spec=spec,
        out_path=args.out, show=args.show,
    )


if __name__ == "__main__":
    main()
