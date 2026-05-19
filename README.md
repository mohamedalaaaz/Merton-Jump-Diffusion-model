1 of 4 — Jump detection
Lee-Mykland filter
Separating real crashes from normal noise
def lee_mykland(r, significance=0.001): W = max(10, min(22, n // 8)) # adaptive window lv = pd.Series(r).rolling(W).std() # local vol z = r / lv # standardise cn = np.sqrt(2 * np.log(n)) crit = cn - (np.log(np.pi) + np.log(np.log(n))) / (2*cn) jump_mask = np.abs(z) > crit # flag jumps
Each day's return is divided by the rolling local volatility to produce a z-score. The critical threshold is derived from extreme-value theory — the expected maximum of n standard normals. Any day where the z-score exceeds this is classified as a jump event, not ordinary diffusion. The significance level was tightened from 0.01 to 0.001 to catch NASDAQ's slow grind crashes that never spike on a single day.
extreme value theory
bipower variation
adaptive window
2 of 4 — Model calibration
Merton MLE
Fitting 5 parameters from daily returns
log-return ~ (μ - ½σ²)dt + σ√dt·Z + Σᵢ Yᵢ
def _merton_nll(pv, r, dt, k_max=20): # Sum k=0..20 Poisson-weighted Gaussian components for k in range(k_max+1): lw = k*log(λ·dt) - λ·dt - log(k!) # Poisson weight mk = (μ - ½σ²)·dt + k·μⱼ # mixture mean vk = σ²·dt + k·σⱼ² # mixture var lc[k] = lw + logN(r | mk, vk) # log-sum-exp trick prevents underflow lml = mc + log(Σ exp(lc - mc)) penalty = max(0, log_lam - log(20))**2 * 50
The Merton model says each daily return is a mixture of: normal diffusion (every day) plus random Poisson jumps (rare events). The likelihood sums over all possible numbers of jumps k = 0, 1, 2 … 20, each weighted by the Poisson probability of k jumps occurring in one day. L-BFGS-B optimiser runs 4 restarts to avoid local minima. A soft penalty above λ=20 replaces the hard bound that caused the optimizer to get stuck in the original version.
Poisson mixture
log-sum-exp
L-BFGS-B
5 params: μ σ λ μⱼ σⱼ


3 of 4 — Regime detection
2-state HMM
Bull vs bear — trained with Baum-Welch EM
A = [[0.95, 0.05], # transition matrix (sticky) [0.10, 0.90]] # bull→bear / bear→bull # E-step: forward pass for t in range(1, n): log_alpha[t,s] = logaddexp( log_alpha[t-1,0] + log_A[0,s], log_alpha[t-1,1] + log_A[1,s]) + log_em[t,s] # M-step: update parameters mu_s[s] = (w * obs).sum() / w.sum()
A Hidden Markov Model observes rolling 21-day realised volatility and infers whether the market is in a Bull (low vol) or Bear (high vol) state. The EM algorithm alternates between: E-step — compute the probability of each state at each time using forward-backward recursion, and M-step — update the transition matrix and emission parameters. After 60 iterations it converges on two clearly separated regimes. The final Kelly signal is a probability-weighted blend of the two regime Kellys.
Baum-Welch EM


4 of 4 — Position sizing
Kelly criterion
How much to trade — jump-adjusted
f* = μ / (σ² + λ·(μⱼ² + σⱼ²))
def kelly_fraction(self, rf=0.): # denominator = total variance incl. jump component ev = self.sigma**2 + self.lam*(self.mu_j**2 + self.sigma_j**2) return (self.mu - rf) / ev # Blend by current regime posterior k_blend = p_bull * kelly_bull + p_bear * kelly_bear # Apply quarter-Kelly for safety position = clip(k_blend * 0.25, 0, 0.25)
Standard Kelly uses f* = μ/σ². The Merton version adds jump variance to the denominator — jumps reduce your optimal bet size because they make ruin more likely. The blended Kelly weights bull and bear estimates by the HMM's current posterior probabilities so the signal shifts smoothly as regimes change. Quarter-Kelly (max 25% of capital) is used because full Kelly is theoretically optimal but practically too aggressive for most accounts.
Kelly criterion
jump-adjusted variance
quarter-Kelly
regime blending
forward-backward
sticky transitions
per-regime JD
