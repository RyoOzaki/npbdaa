import numpy as np
from pyhlm import WeakLimitHDPHLM
from word_model import LetterHSMM
import pyhsmm
import warnings
from tqdm import trange
warnings.filterwarnings('ignore')
import time

#%%
def load_datas():
    data = []
    names = np.loadtxt("files.txt", dtype=str)
    files = names
    for name in names:
        data.append(np.loadtxt("DATA/" + name + ".txt"))
    return data

def unpack_durations(dur):
    unpacked = np.zeros(dur.sum())
    d = np.cumsum(dur[:-1])
    unpacked[d-1] = 1.0
    return unpacked

def save_datas(states_list):
    names = np.loadtxt("files.txt", dtype=str)
    for i, s in enumerate(states_list):
        with open("results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq)
        with open("results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq)
        with open("results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored))

#%%
obs_dim = 3
letter_upper = 10
word_upper = 10
model_hypparams = {'num_states': word_upper, 'alpha': 10, 'gamma': 10, 'init_state_concentration': 10}
obs_hypparams = {
    'mu_0':np.zeros(obs_dim),
    'sigma_0':np.identity(obs_dim),
    'kappa_0':0.01,
    'nu_0':obs_dim+2
}
dur_hypparams = {
    'alpha_0':200,
    'beta_0':10
}

#%%
letter_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(letter_upper)]
letter_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(letter_upper)]
dur_distns = [pyhsmm.distributions.PoissonDuration(lmbda=20) for state in range(word_upper)]
length_distn = pyhsmm.distributions.PoissonDuration(alpha_0=30, beta_0=10, lmbda=3)

#%%
letter_hsmm = LetterHSMM(alpha=10, gamma=10, init_state_concentration=10, obs_distns=letter_obs_distns, dur_distns=letter_dur_distns)
model = WeakLimitHDPHLM(model_hypparams, letter_hsmm, dur_distns, length_distn)

#%%
files = np.loadtxt("files.txt", dtype=str)
datas = load_datas()

#%% Pre training.
for d in datas:
    letter_hsmm.add_data(d, trunc=60)
for t in trange(50):
    letter_hsmm.resample_model(num_procs=32)
letter_hsmm.states_list = []

for d in datas:
    # letter_hsmm.add_data(d)
    model.add_data(d, trunc=60, generate=False)

#%%
for t in trange(100):
    st = time.time()
    model.resample_model(num_procs=32)
    print("resample_model:{}".format(time.time() - st))
    save_datas(model.states_list)
    print(model.word_list)
