import numpy as np
import scipy.stats as stats
from scipy.misc import logsumexp

from pybasicbayes.distributions.poisson import Poisson

from pyhsmm.util.general import list_split
from pyhsmm.util.stats import sample_discrete
from pyhsmm.internals.transitions import WeakLimitHDPHMMTransitions
from pyhsmm.internals.initial_state import HMMInitialState
import time

# import rle

class WeakLimitHDPHLM(object):

    def __init__(self, hypparams, letter_hsmm, dur_distns, length_distn):
        self._letter_hsmm = letter_hsmm
        self._length_distn = length_distn#Poisson(alpha_0=30, beta_0=10)
        self._dur_distns = dur_distns
        self._num_states = hypparams['num_states']
        self._letter_num_states = letter_hsmm.num_states
        self._init_state_distn = HMMInitialState(self, init_state_concentration=hypparams["init_state_concentration"])
        hypparams.pop("init_state_concentration")
        self._trans_distn = WeakLimitHDPHMMTransitions(**hypparams)
        self.states_list = []

        self.word_list = [None] * self.num_states
        for i in range(self.num_states):
            self._generate_word_and_set_at(i)
        self.resample_dur_distns()

    @property
    def num_states(self):
        return self._num_states

    @property
    def letter_num_states(self):
        return self._letter_num_states

    @property
    def letter_obs_distns(self):
        return self.letter_hsmm.obs_distns

    @property
    def dur_distns(self):
        return self._dur_distns

    @property
    def letter_dur_distns(self):
        return self.letter_hsmm.dur_distns

    @property
    def init_state_distn(self):
        return self._init_state_distn

    @property
    def trans_distn(self):
        return self._trans_distn

    @property
    def length_distn(self):
        return self._length_distn

    @property
    def letter_hsmm(self):
        return self._letter_hsmm

    def log_likelihood(self):
        return sum([word_state.log_likelihood() for word_state in self.states_list])

    def generate_word(self):
        size = self.length_distn.rvs() or 1
        return self.letter_hsmm.generate_word(size)

    def _generate_word_and_set_at(self, idx):
        self.word_list[idx] = None
        word = self.generate_word()
        while word in self.word_list:
            word = self.generate_word()
        self.word_list[idx] = word

    def add_data(self, data, **kwargs):
        self.states_list.append(WeakLimitHDPHLMStates(self, data, **kwargs))

    def add_word_data(self, data, **kwargs):
        self.letter_hsmm.add_data(data, **kwargs)

    def resample_model(self, num_procs=0):
        times = [0.] * 34
        st = time.time()
        self.resample_states(num_procs=num_procs)
        times[0] = time.time() - st
        st = time.time()
        self.resample_letter_hsmm(num_procs=num_procs)
        times[1] = time.time() - st
        st = time.time()
        self.resample_words()
        times[2] = time.time() - st
        st = time.time()
        self.resample_length_distn()
        self.resample_dur_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self._clear_caches()
        times[3] = time.time() - st

        print("Resample states:{}".format(times[0]))
        print("Resample letter states:{}".format(times[1]))
        print("SIR:{}".format(times[2]))
        print("Resample others:{}".format(times[3]))

    def resample_states(self, num_procs=0):
        if num_procs == 0:
            for word_state in self.states_list:
                word_state.resample()
        else:
            self._joblib_resample_states(self.states_list,num_procs)

    def _joblib_resample_states(self,states_list,num_procs):
        from joblib import Parallel, delayed
        import parallel

        # warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = list_split(
                    [self._get_joblib_pair(s) for s in states_list],
                    num_procs)

            parallel.model = self
            parallel.args = joblib_args

            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_sampled_stateseq_norep_and_durations_censored)(idx)
                            for idx in range(len(joblib_args)))

            for s, (stateseq, stateseq_norep, durations_censored, log_likelihood) in zip(
                    [s for grp in list_split(states_list,num_procs) for s in grp],
                    [seq for grp in raw_stateseqs for seq in grp]):
                s.stateseq, s._stateseq_norep, s._durations_censored, s._normalizer = stateseq, stateseq_norep, durations_censored, log_likelihood

    def _get_joblib_pair(self,states_obj):
        return (states_obj.data,states_obj._kwargs)

    def resample_letter_hsmm(self, num_procs=0):
        self.letter_hsmm.states_list = []
        [word_state.add_word_datas(generate=False) for word_state in self.states_list]
        self.letter_hsmm.resample_states(num_procs=num_procs)
        self.letter_hsmm.resample_parameters()
        [letter_state.reflect_letter_stateseq() for letter_state in self.letter_hsmm.states_list]

    def resample_words(self):
        for word_idx in range(self.num_states):
            hsmm_states = [letter_state for letter_state in self.letter_hsmm.states_list if letter_state.word_idx == word_idx]
            candidates = [tuple(letter_state.stateseq_norep) for letter_state in hsmm_states]
            unique_candidates = list(set(candidates))
            ref_array = np.array([unique_candidates.index(candi) for candi in candidates])
            if len(candidates) == 0:
                self._generate_word_and_set_at(word_idx)
                continue
            elif len(unique_candidates) == 1:
                self.word_list[word_idx] = unique_candidates[0]
                continue
            cache_score = np.empty((len(unique_candidates), len(candidates)))
            likelihoods = np.array([letter_state.log_likelihood() for letter_state in hsmm_states])
            range_tmp = list(range(len(candidates)))

            for candi_idx, candi in enumerate(unique_candidates):
                tmp = range_tmp[:]
                if (ref_array == candi_idx).sum() == 1:
                    tmp.remove(np.where(ref_array == candi_idx)[0][0])
                for tmp_idx in tmp:
                    # print(hsmm_states[tmp_idx].likelihood_block_word(candi))
                    cache_score[candi_idx, tmp_idx] = hsmm_states[tmp_idx].likelihood_block_word(candi)[-1]
            cache_scores_matrix = cache_score[ref_array]
            for i in range_tmp:
                cache_scores_matrix[i, i] = 0.0
            scores = cache_scores_matrix.sum(axis=1) + likelihoods

            assert (np.exp(scores) >= 0).all(), cache_scores_matrix
            sampled_candi_idx = sample_discrete(np.exp(scores))
            self.word_list[word_idx] = candidates[sampled_candi_idx]

        # Merge same letter seq which has different id.
        for i, word in enumerate(self.word_list):
            if word in self.word_list[:i]:
                for word_state in self.states_list:
                    existed_id = self.word_list[:i].index(word)
                    stateseq, stateseq_norep = word_state.stateseq, word_state.stateseq_norep
                    word_state.stateseq[stateseq == i] = existed_id
                    word_state.stateseq_norep[stateseq_norep == i] = existed_id
                    self._generate_word_and_set_at(i)

    def resample_length_distn(self):
        self.length_distn.resample(np.array([len(word) for word in self.word_list]))

    def resample_dur_distns(self):#Do not resample!! This code only update the parameter of duration distribution of word.
        letter_lmbdas = np.array([letter_dur_distn.lmbda for letter_dur_distn in self.letter_dur_distns])
        for word, dur_distn in zip(self.word_list, self.dur_distns):
            dur_distn.lmbda = np.sum(letter_lmbdas[list(word)])

    def resample_trans_distn(self):
        self.trans_distn.resample([word_state.stateseq_norep for word_state in self.states_list])

    def resample_init_state_distn(self):
        self.init_state_distn.resample(np.array([word_state.stateseq_norep[0] for word_state in self.states_list]))

    def _clear_caches(self):
        for word_state in self.states_list:
            word_state.clear_caches()

    def params(self):
        self.trans_distn.params()

class WeakLimitHDPHLMStates(object):

    def __init__(self, model, data=None, trunc=None, generate=True, initialize_from_prior=True):
        self.model = model
        self.data = data
        self.T = T = len(data)
        self.trunc = trunc
        self._stateseq = np.empty(T, dtype=np.int32)
        self._stateseq_norep = None
        self._durations_censored = None
        self._letter_stateseq = np.empty(T, dtype=np.int32)
        self._kwargs = dict(trunc=trunc)
        if generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()
        self.clear_caches()

    def generate_states(self):
        raise NotImplementedError

    @property
    def stateseq(self):
        return self._stateseq

    @stateseq.setter
    def stateseq(self, stateseq):
        self._stateseq = stateseq

    @property
    def letter_stateseq(self):
        return self._letter_stateseq

    @letter_stateseq.setter
    def letter_stateseq(self, letter_stateseq):
        self._letter_stateseq = letter_stateseq

    @property
    def stateseq_norep(self):
        if self._stateseq_norep is None:
            self._stateseq_norep, self._durations_censored  = rle(self.stateseq)
        return self._stateseq_norep

    @property
    def durations_censored(self):
        if self._durations_censored is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._durations_censored

    # Be care full!!!!
    # This method return the log likelihood which before resampling this model.
    def log_likelihood(self):
        if self._normalizer is None:
            _, _, normalizerl = self.messages_backwards()
            self._normalizer = normalizerl
        return self._normalizer

    @property
    def pi_0(self):
        return self.model.init_state_distn.pi_0

    @property
    def aDl(self):
        if self._aDl is None:
            aDl = np.empty((self.T,self.model.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.model.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._aDl = aDl
        return self._aDl

    @property
    def alDl(self):
        if self._alDl is None:
            alDl = np.empty((self.T,self.model.letter_num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.model.letter_dur_distns):
                alDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._alDl = alDl
        return self._alDl

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = np.empty((self.data.shape[0], self.model._letter_num_states))
            for idx, obs_distn in enumerate(self.model.letter_obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(self.data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.0
            self._aBl = aBl
        return self._aBl

    @property
    def log_trans_matrix(self):
        if self._log_trans_matrix is None:
            self._log_trans_matrix = np.log(self.model.trans_distn.trans_matrix)
        return self._log_trans_matrix

    def resample(self):
        self.clear_caches()
        betal, betastarl, normalizerl = self.messages_backwards()
        self._normalizer = normalizerl
        self.sample_forwards(betal, betastarl)

    def messages_backwards(self):
        aDl = self.aDl
        log_trans_matrix = self.log_trans_matrix
        T = self.T
        pi_0 = self.pi_0
        trunc = self.trunc if self.trunc is not None else T
        betal = np.zeros((T, self.model.num_states), dtype=np.float64)
        betastarl = np.zeros((T, self.model.num_states), dtype=np.float64)
        normalizerl = 0.0

        for t in range(T-1, -1, -1):
            betastarl[t] = np.logaddexp.reduce(
                betal[t:t+trunc] + self.cumulative_likelihoods(t, t+trunc) + aDl[:min(trunc, T-t)],
                axis=0
            )
            betal[t-1] = np.logaddexp.reduce(betastarl[t] + log_trans_matrix, axis=1)
        betal[-1] = 0.0
        normalierl = np.logaddexp.reduce(betastarl[0] + np.log(pi_0))
        return betal, betastarl, normalizerl

    def cumulative_likelihoods(self, start, stop):
        T = min(self.T, stop)
        tsize = T - start
        cum_like = np.empty((tsize, self.model.num_states), dtype=np.float64)

        for state, word in enumerate(self.model.word_list):
            cum_like[:, state] = self.likelihood_block_word(start, stop, word)

        return cum_like

    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl
        alDl = self.alDl
        len_word = len(word)
        alphal = np.ones((tsize, len_word), dtype=np.float64) * -np.inf

        if tsize-len_word+1 <= 0:
            return alphal[:, -1]

        cumsum_aBl = np.empty(tsize-len_word+1, dtype=np.float64)
        alphal[:tsize-len_word+1, 0] = np.cumsum(aBl[start:start+tsize-len_word+1, word[0]]) + alDl[:tsize-len_word+1, word[0]]
        cache_range = range(tsize - len_word + 1)
        for j, l in enumerate(word[1:]):
            cumsum_aBl[:] = 0.0
            for t in cache_range:
                cumsum_aBl[:t+1] += aBl[start+t+j+1, l]
                alphal[t+j+1, j+1] = np.logaddexp.reduce(cumsum_aBl[:t+1] + alDl[t::-1, l] + alphal[j:t+j+1, j])
        return alphal[:, -1]

    def sample_forwards(self, betal, betastarl):
        T = self.T
        aD = np.exp(self.aDl)
        log_trans_matrix = self.log_trans_matrix
        stateseq = self._stateseq[:]
        stateseq[:] = -1
        letter_stateseq = self._letter_stateseq[:]
        letter_stateseq[:] = -1
        stateseq_norep = []
        durations_censored = []
        t = 0
        nextstate_unsmoothed = self.pi_0
        while t < T:
            logdomain = betastarl[t] - betastarl[t].max()
            nextstate_dist = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_dist == 0.).all():
                nextstate_dist = np.exp(logdomain)

            state = sample_discrete(nextstate_dist)
            durprob = np.random.random()
            # dur = len(self.model.word_list[state])
            cache_mess_term = np.exp(self.likelihood_block_word(t, T, self.model.word_list[state]) + betal[t:T, state] - betastarl[t, state])

            dur = 0
            while durprob > 0 and t+dur < T:
                # p_d_prior = aD[dur, state] if t + dur < T else 1.
                p_d_prior = aD[dur, state]
                assert not np.isnan(p_d_prior)
                assert p_d_prior >= 0

                p_d = cache_mess_term[dur] * p_d_prior
                assert not np.isnan(p_d)
                durprob -= p_d
                dur += 1

            assert dur > 0
            assert dur >= len(self.model.word_list[state])
            stateseq[t:t+dur] = state
            nextstate_unsmoothed = nextstate_dist[state]
            t += dur

            stateseq_norep.append(state)
            durations_censored.append(dur)
        self._stateseq_norep = np.array(stateseq_norep, dtype=np.int32)
        self._durations_censored = np.array(durations_censored, dtype=np.int32)

    def clear_caches(self):
        self._aBl = None
        self._aDl = None
        self._alDl = None
        self._log_trans_matrix = None

    def add_word_datas(self, **kwargs):
        s = self.stateseq_norep
        d = self.durations_censored
        dc = np.concatenate(([0], d)).cumsum()
        for i, word_idx in enumerate(s):
            self.model.add_word_data(self.data[dc[i]:dc[i+1]], hlmstate=self, word_idx=word_idx, d0=dc[i], d1=dc[i+1], **kwargs)
