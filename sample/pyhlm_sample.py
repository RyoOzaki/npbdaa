import os
import numpy as np
from pyhlm.model import WeakLimitHDPHLM, WeakLimitHDPHLMPython
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.word_model import LetterHSMM
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

def save_stateseq(model):
    # Save sampled states sequences.
    names = np.loadtxt("files.txt", dtype=str)
    for i, s in enumerate(model.states_list):
        with open("results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq)
        with open("results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq)
        with open("results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored))

def save_params(itr_idx, model):
    with open("parameters/ITR_{0:04d}.txt".format(itr_idx), "w") as f:
        f.write(str(model.params))

def save_loglikelihood(model):
    with open("summary_files/log_likelihood.txt", "a") as f:
        f.write(str(model.log_likelihood()) + "\n")

def save_resample_times(resample_time):
    with open("summary_files/resample_times.txt", "a") as f:
        f.write(str(resample_time) + "\n")


#%%
if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('parameters'):
    os.mkdir('parameters')

if not os.path.exists('summary_files'):
    os.mkdir('summary_files')

#%%
thread_num = 4
pre_train_iter = 1
train_iter = 100
trunc = 60
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
    letter_hsmm.add_data(d, trunc=trunc)
for t in trange(pre_train_iter):
    letter_hsmm.resample_model(num_procs=thread_num)
letter_hsmm.states_list = []

#%%
print("Add datas...")
for d in datas:
    model.add_data(d, trunc=trunc, generate=False)
model.resample_states(num_procs=thread_num)
# # or
# for d in datas:
#     model.add_data(d, trunc=trunc, initialize_from_prior=False)
print("Done!")

#%% Save init params and pyper params
with open("parameters/hypparams.txt", "w") as f:
    f.write(str(model.hypparams))
save_params(0, model)
save_loglikelihood(model)

#%%
for t in trange(train_iter):
    st = time.time()
    model.resample_model(num_procs=thread_num)
    resample_model_time = time.time() - st
    save_stateseq(model)
    save_loglikelihood(model)
    save_params(t+1, model)
    save_resample_times(resample_model_time)
    print(model.word_list)
    print(model.word_counts())
    print("log_likelihood:{}".format(model.log_likelihood()))
    print("resample_model:{}".format(resample_model_time))
