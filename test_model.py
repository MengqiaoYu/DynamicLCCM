import numpy as np
from hmmlearn import hmm
import MixtureHMM

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"


# generate multiple sequences of sample
def sampling(N=1000, num_states = 2, num_timesteps = 12):
    model = hmm.MultinomialHMM(n_components=num_states)
    model.startprob_ = np.array([0.4, 0.6])
    model.transmat_ = np.array([[0.7, 0.3],
                                [0.1, 0.9]])
    model.emissionprob_ = np.array([[0.4, 0.6],
                                   [0.8, 0.2]])
    seq = []
    for i in range(N):
        X, Z = model.sample(n_samples=num_timesteps)
        seq.append(X.reshape(-1))
    return seq

sample = sampling()

MHMM = MixtureHMM.MixtureHMM(num_states=2)
MHMM.train(obs_seq=sample, cutoff_value=1e-3)