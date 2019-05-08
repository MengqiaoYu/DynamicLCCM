import numpy as np
from hmmlearn import hmm
import MixtureHMM
import logging
from os import listdir
import pandas as pd

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"


# generate multiple sequences of sample
def sampling_mixture(N=1000, num_states = 2, num_timesteps = 12):
    """test for MixtureHMM"""
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
    # print(seq[10])
    return seq

# test MixtureHMM (have calibrated with benchmark model)
def train_mixture():
    print("This section is for training MixtureHMM example")
    sample = sampling_mixture()
    MHMM = MixtureHMM.MixtureHMM(num_states=2)
    MHMM.train(obs_seq=sample, cutoff_value=1e-3, max_iter=500)

# test HeteroMixtureHMM
def train_hetero():
    """"We don't need sampling anymore, we directly feed our own dataset and test the model."""
    data_dir = '/Users/MengqiaoYu/Desktop/Research/WholeTraveler/Data/model/'
    print("Load the formatted files in dir: %s into model." %data_dir)
    data_model = []
    for f in listdir(data_dir):
        if f.startswith('.'):
            continue
        data_ind = pd.read_csv(data_dir + f)
        data_model.append(data_ind[['used_public', 'used_own', 'move', 'age', 'employ']].values)

    # full_header =list(data_ind)
    header = ['used_public', 'used_own', 'move', 'age', 'employ']
    choices = ['used_public', 'used_own']
    trans_cov = ['move', 'age', 'employ']

    HHMM = MixtureHMM.HeteroMixtureHMM(num_states=2)
    HHMM.set_dataframe(samples=data_model,
                       header=header,
                       choices=choices,
                       trans_cov=trans_cov)
    HHMM.train(cutoff_value=1e-3, max_iter=200)

train_hetero()