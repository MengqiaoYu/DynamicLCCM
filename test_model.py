import numpy as np
from hmmlearn import hmm
from hmmlearn import base
import MixtureHMM

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"


# generate multiple sequences of sample
def sampling(N=1000, num_states = 2, num_timesteps = 12):
    model = hmm.MultinomialHMM(n_components=num_states)
    model.startprob_ = np.array([0.4, 0.6])
    model.transmat_ = np.array([[0.7, 0.3],
                                [0.1, 0.9]])
    model.emissionprob_ = np.array([[0.3, 0.7],
                                   [0.65, 0.35]])
    seq = []
    for i in range(N):
        X_1, Z = model.sample(n_samples=num_timesteps)
        X_2 = np.copy(X_1)
        emis_prob_2 = np.array([[0.8, 0.2],
                                [0.1, 0.9]])
        for i, z in enumerate(Z):
            ram_no = np.random.rand(1)[0]
            if ram_no <= emis_prob_2[z][0]:
                X_2[i] = 0
            else:
                X_2[i] = 1

        X = np.concatenate((X_1, X_2), axis=1)
        seq.append(X)
    print(seq[10])
    return seq

sample = sampling()

MHMM = MixtureHMM.MixtureHMM(num_states=2)
MHMM.train(obs_seq=sample, cutoff_value=1e-3, max_iter=500)
