import numpy as np

def generate_hmm_data(mus, sigmas, A, pi, N):
    K = len(mus)
    D = mus[0].shape[0]
    
    states = np.zeros(N, dtype=int)
    observations = np.zeros((N, D))
    
    states[0] = np.random.choice(K, p=pi)
    observations[0] = np.random.multivariate_normal(mus[states[0]], sigmas[states[0]])
    
    for t in range(1, N):
        states[t] = np.random.choice(K, p=A[states[t-1]])
        observations[t] = np.random.multivariate_normal(mus[states[t]], sigmas[states[t]])
    
    return observations, states

K = 3
D = 3

mus = np.array([
    [-0.10,  0.10,  0.10],
    [ 0.00,  0.00,  0.00],
    [ 0.10, -0.10, -0.10],
])

sigmas = np.array([
    [[0.0001, 0.0000, 0.0000],
     [0.0000, 0.0001, 0.0000],
     [0.0000, 0.0000, 0.0001]],

    [[0.0001, 0.0000, 0.0000],
     [0.0000, 0.0001, 0.0000],
     [0.0000, 0.0000, 0.0001]],

    [[0.0001, 0.0000, 0.0000],
     [0.0000, 0.0001, 0.0000],
     [0.0000, 0.0000, 0.0001]],
])

A = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90],
])

pi = np.array([1/3, 1/3, 1/3])

def synthetic_data():
    obs, states = generate_hmm_data(mus, sigmas, A, pi, N=730)
    return obs, states

if __name__ == "__main__":
    obs, states = synthetic_data()
    print(obs.shape[1])

