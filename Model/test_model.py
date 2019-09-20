import numpy as np
import MixtureHMM
from os import listdir
import pandas as pd

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"


# generate multiple sequences of sample
def sampling_mixture(N=1000, num_states = 2, num_timesteps = 12):
    """test for MixtureHMM"""
    from hmmlearn import hmm
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
    MHMM.train_MixtureHMM(obs_seq=sample, cutoff_value=1e-3, max_iter=500)

# test HeteroMixtureHMM
def train_hetero():
    """"We don't need sampling anymore, we directly feed our own dataset and test the model."""
    data_dir = '/Users/MengqiaoYu/Desktop/Research/WholeTraveler/Data/model/'

    # used_public, used_ridehail, used_own, used_walkbike, numcars_cat
    choices = ['used_public', 'used_own', 'used_ridehail', 'used_walkbike', 'numcars_cat']

    # child, move, edu, partner, youngchild, employ, school, ue, iu, age_rescaled
    trans_cov = ['move', 'partner', 'child', 'employ', 'ue', 'iu']
    print(trans_cov)

    # full_header =list(data_ind)
    header = choices + trans_cov

    print("Load the formatted files in dir: %s into model." %data_dir)
    data_model = []
    for f in listdir(data_dir):
        if f.startswith('.'):
            continue
        data_ind = pd.read_csv(data_dir + f)
        data_model.append(data_ind[header].values)
        if len(data_model) % 200 == 0:
            print("\t Processed %d files." %len(data_model))

    # Set the model
    print("Start training the model...")
    HHMM = MixtureHMM.HeteroMixtureHMM(num_states=3)
    HHMM.gen_train_data(data=data_model,
                        header=header,
                        choices_header=choices,
                        trans_cov_header=trans_cov)
    HHMM.train_HeteroMixtureHMM(cutoff_value=1e-3, max_iter=100, print_std=True, plot_trend=True)
    print("-----------Training ends-----------")

    # Prediction


if __name__ == '__main__':
    # train_mixture()
    train_hetero()
