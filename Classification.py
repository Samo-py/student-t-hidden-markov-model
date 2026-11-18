import numpy as np
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

from agregate import data_fetch

dataframe = yf.download("BTC-USD", auto_adjust=True)[["Close"]]
dataframe.columns = ["price_change"]
dataframe.index.name = None

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
### features
D = observations.shape[1]
print(D)
def mult_pdf(x, mu, Sigma):
    x = np.asarray(x)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    k = mu.shape[0]
    diff = x - mu

    det = np.linalg.det(Sigma)
    inv = np.linalg.solve(Sigma, diff)

    norm_const = 1 / np.sqrt((2 * np.pi) ** k * det)
    exponent = -0.5 * diff @ inv

    return norm_const * np.exp(exponent)

def log_mult_pdf(x, mu, Sigma):
    x = np.asarray(x)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    k = mu.shape[0]
    diff = x - mu

    _, log_det = np.linalg.slogdet(Sigma)
    inv = np.linalg.solve(Sigma, diff)

    return -0.5 * (k * np.log(2 * np.pi) + log_det + diff @ inv)

log_coefs = np.log(np.ones(K) / K)

# A = np.array([
#     [0.8, 0.1, 0.1],
#     [0.1, 0.8, 0.1],
#     [0.1, 0.1, 0.8]
# ])

A = np.array([
    [0.98, 0.01, 0.01],
    [0.01, 0.98, 0.01],
    [0.01, 0.01, 0.98]
])



# global_mean = observations.mean(axis=0)
# mus = np.array([global_mean + np.random.randn(D) * 0.01 for _ in range(K)])


# global_cov = np.cov(observations.T)
# sigmas = np.array([global_cov.copy() for _ in range(K)])

global_mean = observations.mean(axis=0)
global_cov = np.cov(observations.T)

mus = np.zeros((K, D))
mus[0] = np.array([-1.5,  0.0,  1.5, -1.0,  1.0, -0.5])
mus[1] = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
mus[2] = np.array([ 1.5,  0.0, -1.0,  1.0, -1.0,  0.5])

sigmas = np.array([
    global_cov * 2.0,
    global_cov * 0.5,
    global_cov * 1.0,
])

iterations = 50
tol = 1e-4
prev_log_likelihood = -np.inf

### Baum-Welch

for iteration in range(iterations):

    log_emissions = np.zeros((N, K))
    for t in range(N):
        for k in range(K):
            log_emissions[t, k] = log_mult_pdf(observations[t], mus[k], sigmas[k])

    ### Expectation
    ### Forward-Backward Alpha/Beta

    log_alpha = np.zeros((N, K))

    for k in range(K):
        log_alpha[0, k] = log_coefs[k] + log_mult_pdf(observations[0], mus[k], sigmas[k])

    # for t in range(1, N):
    #     for j in range(K):
    #         log_terms = log_alpha[t-1, :] + np.log(A[:, j])
    #         log_sum = np.logaddexp.reduce(log_terms)

    #         log_alpha[t, j] = log_mult_pdf(observations[t], mus[j], sigmas[j]) + log_sum

    # log_likelihood = np.logaddexp.reduce(log_alpha[N-1, :])

    log_A = np.log(A)

    for t in range(1, N):
        log_terms = log_alpha[t-1, :, None] + log_A
        log_sum = np.logaddexp.reduce(log_terms, axis=0)
        log_alpha[t] = log_emissions[t] + log_sum

    log_likelihood = np.logaddexp.reduce(log_alpha[N-1, :])

    log_beta = np.zeros((N, K))
    log_beta[N-1, :] = 0

    # for t in range(N-2, -1, -1):
    #     for i in range(K):
    #         log_terms = np.log(A[i, :]) + np.array([log_mult_pdf(observations[t+1], mus[j], sigmas[j]) for j in range(K)]) + log_beta[t+1, :]
    #         log_beta[t, i] = np.logaddexp.reduce(log_terms)

    for t in range(N-2, -1, -1):
        log_terms = log_A + log_beta[t+1, None, :] + log_emissions[t+1, None :]
        log_beta[t, :] = np.logaddexp.reduce(log_terms, axis=1)


    ### Posterior probabilities Gamma/Xi

    log_gamma = log_alpha + log_beta
    for t in range(N):
        log_sum = np.logaddexp.reduce(log_gamma[t, :])
        log_gamma[t, :] -= log_sum
    gamma = np.exp(log_gamma)

    # log_xi = np.zeros((N-1, K, K))
    # for t in range(N-1):
    #     for i in range(K):
    #         for j in range(K):
    #             log_xi[t, i, j] = log_alpha[t, i] + np.log(A[i, j]) + log_beta[t+1, j] + log_mult_pdf(observations[t+1], mus[j], sigmas[j])
    #     log_sum = np.logaddexp.reduce(log_xi[t, :, :].flatten())
    #     log_xi[t, :, :] -= log_sum

    # log_emissions = np.zeros((N, K))
    # for t in range(N):
    #     for k in range(K):
    #         log_emissions[t, k] = log_mult_pdf(observations[t], mus[k], sigmas[k])

    log_xi = np.zeros((N-1, K, K))
    for t in range(N - 1):    
        log_xi[t] = log_alpha[t, :, None] + np.log(A[None, :, :]) + log_beta[t+1, None, :] + log_emissions[t+1, None, :]
    
        log_sum = np.logaddexp.reduce(log_xi[t].flatten())
        log_xi[t] -= log_sum

    xi = np.exp(log_xi)

    ### Maximization

    log_coefs = log_gamma[0, :]

    # for i in range(K):
    #     for j in range(K):
    #         A[i,j] = xi[:, i, j].sum()/ gamma[:-1,i].sum()
    #         A[i, j] = np.clip(A[i,j], 0.005, 0.995)

    A = xi.sum(axis=0)/ gamma[:-1].sum(axis=0)[:, None]
    A = np.clip(A, 0.001, 0.999)

    for k in range(K):
        mus[k] = (gamma[:, k, None] * observations).sum(axis=0) / gamma[:, k].sum()

    for k in range(K):
        diff = observations - mus[k]
        sigmas[k] = (gamma[:, k, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)/gamma[:,k].sum()
        sigmas[k] += np.eye(D) * 1e-5

    if iteration > 0:
        improvement = log_likelihood - prev_log_likelihood
        print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}, improvement = {improvement:.4f}")
        
        if improvement < tol:
            print(f"Converged at iteration {iteration}")
            break
    else:
        print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")

    prev_log_likelihood = log_likelihood

### Virterbi


log_delta = np.zeros((N, K)) - np.inf
psi = np.zeros((N, K), dtype=int)

for k in range(K):
    log_delta[0, k] = log_coefs[k] + log_mult_pdf(observations[0], mus[k], sigmas[k])

for t in range(1, N):
    for j in range(K):
        best_log = -np.inf
        best_i = 0

        for i in range(K):
            log_val = log_delta[t-1, i] + np.log(max(A[i, j], 1e-100))
            if log_val >= best_log:

                best_log = log_val
                best_i = i

        emission = max(log_mult_pdf(observations[t], mus[j], sigmas[j]), -100)
        log_delta[t, j] = best_log + emission
        psi[t, j] = best_i

### Virterbi path

best_path = np.zeros(N, dtype=int)
best_path[N-1] = np.argmax(log_delta[N-1])

for t in range(N-2, -1, -1):
    best_path[t] = psi[t+1, best_path[t+1]]


print(mus)
print(sigmas)
print(A)






time = dataframe[start:end].index.astype("str")

colors = ["#FFDDDD", "#DDFFDD", "#DDDDFF"]
regime_labels = ["Regime 0", "Regime 1", "Regime 2"]

fig, ax = plt.subplots(figsize=(12, 4))

regime_start = 0
for t in range(1, N):
    if best_path[t] != best_path[t-1] or t == N-1:
        ax.axvspan(regime_start, t, color=colors[best_path[t-1]], alpha=0.3)
        regime_start = t

ax.plot(time, dataframe[start:end]["price_change"], color='tab:blue', linewidth=2)
for t in range(1, N):
    if best_path[t] != best_path[t-1]:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.5)


ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_xticks(time[::len(time) // 5])
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=colors[i], alpha=0.3, label=regime_labels[i]) for i in range(3)]
ax.legend(handles=legend_patches, loc='upper right')


plt.tight_layout()
plt.savefig("normal.png")
plt.show()