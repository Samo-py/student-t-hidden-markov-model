import numpy as np
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.special import digamma, gammaln

from agregate import data_fetch

dataframe = yf.download("BTC-USD", auto_adjust=True)[["Close"]]
dataframe.columns = ["price_change"]
dataframe.index.name = None

start = "2021-11-09"
end = "2022-11-21"

start = end
end = datetime.strftime(datetime.today().date(), format="%Y-%m-%d")

features = asyncio.run(data_fetch(start, end))
observations = np.array(features)

K = 3
N = len(observations)
D = observations.shape[1]
print(f"D={D}, N={N}")


# ---------------------------------------------------------------------------
# Multivariate Student-t  log-pdf
#
#   p(x | mu, Sigma, nu) = C * |Sigma|^{-1/2} * (1 + delta/nu)^{-(nu+D)/2}
#   where  delta = (x-mu)^T Sigma^{-1} (x-mu)
#
# The per-sample Mahalanobis weight used in the M-step is:
#   u = (nu + D) / (nu + delta)
# ---------------------------------------------------------------------------

def log_student_t_pdf(x, mu, Sigma, nu):
    """Log-pdf of the multivariate Student-t distribution."""
    x   = np.asarray(x, dtype=float)
    mu  = np.asarray(mu, dtype=float)
    nu  = float(nu)

    diff      = x - mu
    _, log_det = np.linalg.slogdet(Sigma)
    inv_diff  = np.linalg.solve(Sigma, diff)
    delta     = diff @ inv_diff                       # Mahalanobis^2

    log_C = (gammaln((nu + D) / 2)
             - gammaln(nu / 2)
             - 0.5 * D * np.log(nu * np.pi)
             - 0.5 * log_det)

    return log_C - 0.5 * (nu + D) * np.log(1.0 + delta / nu)


def mahalanobis_sq(x, mu, Sigma):
    """(x-mu)^T Sigma^{-1} (x-mu)"""
    diff = x - mu
    return diff @ np.linalg.solve(Sigma, diff)


def u_weight(delta, nu):
    """E-step weight for Student-t M-step: E[latent precision u | x]."""
    return (nu + D) / (nu + delta)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

log_coefs = np.log(np.ones(K) / K)

A = np.array([
    [0.98, 0.01, 0.01],
    [0.01, 0.98, 0.01],
    [0.01, 0.01, 0.98],
])

global_mean = observations.mean(axis=0)
global_cov  = np.cov(observations.T)

mus = np.zeros((K, D))
mus[0] = np.array([-1.5,  0.0,  1.5, -1.0,  1.0, -0.5])
mus[1] = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
mus[2] = np.array([ 1.5,  0.0, -1.0,  1.0, -1.0,  0.5])

sigmas = np.array([
    global_cov * 2.0,
    global_cov * 0.5,
    global_cov * 1.0,
])

# Degrees of freedom per state — keep > 2 so variance is finite
nus = np.array([5.0, 5.0, 5.0])

iterations = 200
tol        = 1e-4
prev_log_likelihood = -np.inf

# ---------------------------------------------------------------------------
# Baum-Welch with Student-t emissions
# ---------------------------------------------------------------------------

for iteration in range(iterations):

    # ---- pre-compute log-emissions and Mahalanobis distances ---------------
    log_emissions = np.zeros((N, K))
    delta_mat     = np.zeros((N, K))   # Mahalanobis^2  (needed for M-step)

    for t in range(N):
        for k in range(K):
            delta_mat[t, k]     = mahalanobis_sq(observations[t], mus[k], sigmas[k])
            log_emissions[t, k] = log_student_t_pdf(observations[t], mus[k], sigmas[k], nus[k])

    # ---- Forward pass -------------------------------------------------------
    log_A     = np.log(A)
    log_alpha = np.zeros((N, K))

    for k in range(K):
        log_alpha[0, k] = log_coefs[k] + log_emissions[0, k]

    for t in range(1, N):
        log_terms     = log_alpha[t-1, :, None] + log_A
        log_sum       = np.logaddexp.reduce(log_terms, axis=0)
        log_alpha[t]  = log_emissions[t] + log_sum

    log_likelihood = np.logaddexp.reduce(log_alpha[N-1, :])

    # ---- Backward pass ------------------------------------------------------
    log_beta       = np.zeros((N, K))
    log_beta[N-1]  = 0.0

    for t in range(N-2, -1, -1):
        log_terms    = log_A + log_beta[t+1, None, :] + log_emissions[t+1, None, :]
        log_beta[t]  = np.logaddexp.reduce(log_terms, axis=1)

    # ---- Posterior  γ / ξ ---------------------------------------------------
    log_gamma = log_alpha + log_beta
    for t in range(N):
        log_sum         = np.logaddexp.reduce(log_gamma[t, :])
        log_gamma[t]   -= log_sum
    gamma = np.exp(log_gamma)          # (N, K)

    log_xi = np.zeros((N-1, K, K))
    for t in range(N-1):
        log_xi[t]  = (log_alpha[t, :, None]
                      + log_A
                      + log_beta[t+1, None, :]
                      + log_emissions[t+1, None, :])
        log_xi[t] -= np.logaddexp.reduce(log_xi[t].flatten())

    xi = np.exp(log_xi)

    # ---- M-step weights for Student-t  u_tk = (nu_k+D)/(nu_k+delta_tk) ----
    # shape (N, K)
    u_mat = np.zeros((N, K))
    for k in range(K):
        u_mat[:, k] = u_weight(delta_mat[:, k], nus[k])

    # ---- Maximisation -------------------------------------------------------
    # initial state
    log_coefs = log_gamma[0, :]

    # transition matrix
    # A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
    # A = np.clip(A, 0.001, 0.999)

    ### dirichlet prior
    alpha_diag = 1000.0
    A_counts = xi.sum(axis=0)
    A_counts += alpha_diag * np.eye(K)
    A = A_counts / A_counts.sum(axis=1, keepdims=True)
    A = np.clip(A, 0.001, 0.999)

    # means  (gamma and u both weight the observations)
    for k in range(K):
        w           = gamma[:, k] * u_mat[:, k]          # (N,)
        mus[k]      = (w[:, None] * observations).sum(axis=0) / w.sum()

    # covariances
    for k in range(K):
        w      = gamma[:, k] * u_mat[:, k]
        diff   = observations - mus[k]                    # (N, D)
        sigmas[k] = ((w[:, None, None]
                      * diff[:, :, None]
                      * diff[:, None, :]).sum(axis=0)
                     / gamma[:, k].sum())
        sigmas[k] += np.eye(D) * 1e-5

    # degrees of freedom  (fixed-point update, one Newton step per state)
    # Equation:  -psi(nu/2) + log(nu/2) + 1 + (1/N_k) sum_t gamma_tk [log u_tk - u_tk] + psi((nu+D)/2) - log((nu+D)/2) = 0
    for k in range(K):
        N_k   = gamma[:, k].sum()
        u_k   = u_mat[:, k]
        nu_k  = nus[k]

        # expected log u and expected u (weighted by gamma)
        E_log_u = (gamma[:, k] * (np.log(u_k + 1e-300))).sum() / N_k
        E_u     = (gamma[:, k] * u_k).sum() / N_k

        # fixed-point score (gradient of the nu log-likelihood)
        def score(nu):
            return (- digamma(nu / 2)
                    + np.log(nu / 2)
                    + 1.0
                    + E_log_u - E_u
                    + digamma((nu + D) / 2)
                    - np.log((nu + D) / 2))

        # simple bisection search  (nu must stay in (2, 100))
        lo, hi = 2.01, 100.0
        if score(lo) * score(hi) < 0:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if score(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            nus[k] = 0.5 * (lo + hi)
        # else: leave nu unchanged this iteration

    # ---- Convergence check --------------------------------------------------
    if iteration > 0:
        improvement = log_likelihood - prev_log_likelihood
        print(f"Iter {iteration:3d}: log-lik = {log_likelihood:.2f}  Δ = {improvement:.4f}  nus = {nus.round(2)}")
        if improvement < tol:
            print(f"Converged at iteration {iteration}")
            break
    else:
        print(f"Iter {iteration:3d}: log-lik = {log_likelihood:.2f}  nus = {nus.round(2)}")

    prev_log_likelihood = log_likelihood

print("\nMeans:\n",  mus)
print("\nDegrees of freedom:", nus)
print("\nTransition matrix:\n", A.round(4))

# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------

log_delta       = np.full((N, K), -np.inf)
psi             = np.zeros((N, K), dtype=int)

for k in range(K):
    log_delta[0, k] = log_coefs[k] + log_student_t_pdf(observations[0], mus[k], sigmas[k], nus[k])

for t in range(1, N):
    for j in range(K):
        candidates  = log_delta[t-1] + np.log(np.maximum(A[:, j], 1e-100))
        best_i      = int(np.argmax(candidates))
        emission    = max(log_student_t_pdf(observations[t], mus[j], sigmas[j], nus[j]), -1e6)
        log_delta[t, j] = candidates[best_i] + emission
        psi[t, j]       = best_i

best_path       = np.zeros(N, dtype=int)
best_path[N-1]  = np.argmax(log_delta[N-1])
for t in range(N-2, -1, -1):
    best_path[t] = psi[t+1, best_path[t+1]]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

time   = dataframe[start:end].index.astype("str")
colors = ["#FFDDDD", "#DDFFDD", "#DDDDFF"]
labels = [f"Regime {i}  (ν={nus[i]:.1f})" for i in range(K)]

fig, ax = plt.subplots(figsize=(12, 4))

regime_start = 0
for t in range(1, N):
    if best_path[t] != best_path[t-1] or t == N-1:
        ax.axvspan(regime_start, t, color=colors[best_path[t-1]], alpha=0.3)
        regime_start = t

ax.plot(time, dataframe[start:end]["price_change"], color="tab:blue", linewidth=2)
for t in range(1, N):
    if best_path[t] != best_path[t-1]:
        ax.axvline(t, color="gray", linestyle="--", alpha=0.5)

ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_xticks(time[::max(1, len(time) // 5)])
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=colors[i], alpha=0.3, label=labels[i]) for i in range(K)]
ax.legend(handles=legend_patches, loc="upper right")
plt.tight_layout()
plt.savefig("student-t")
plt.show()