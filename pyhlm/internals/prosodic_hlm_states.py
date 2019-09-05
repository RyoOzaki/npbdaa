import numpy as np

from pyhsmm.util.stats import sample_discrete

from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStatesPython

class WeakLimitProsodicHDPHLMStatesPython(WeakLimitHDPHLMStatesPython):

    def __init__(self, model, data=None, prosody_data=None, **kwargs):
        self.prosody_data = prosody_data
        super(WeakLimitProsodicHDPHLMStatesPython, self).__init__(model, data=data, **kwargs)

    @property
    def apBl(self):
        if self._apBl is None:
            apBl = np.empty((self.prosody_data.shape[0], 2)) #self.model._letter_num_states
            for idx, obs_distn in enumerate(self.model.prosody_distns):
                apBl[:,idx] = obs_distn.log_likelihood(self.prosody_data).ravel()
            apBl[np.isnan(apBl).any(1)] = 0.0
            self._apBl = apBl
        return self._apBl

    def prosody_likelihood_block_word(self, start, stop, state):
        T = min(self.T, stop)
        tsize = T - start
        apBl = self.apBl[start:T]
        aDl = self.aDl[:tsize, state]
        palphal = np.ones((tsize, ), dtype=np.float64) * -np.inf

        return prosodic_hlm_messages_forwards_log(apBl, aDl, palphal)

    def cumulative_prosody_likelihoods(self, start, stop):
        T = min(self.T, stop)
        tsize = T - start
        cum_like = np.empty((tsize, self.model.num_states), dtype=np.float64)

        for state, word in enumerate(self.model.word_list):
            cum_like[:, state] = self.prosody_likelihood_block_word(start, stop, state)
        return cum_like

    def messages_backwards(self):
        aDl = self.aDl
        log_trans_matrix = self.log_trans_matrix
        T = self.T
        pi_0 = self.pi_0
        trunc = self.trunc if self.trunc is not None else T
        betal = np.zeros((T, self.model.num_states), dtype=np.float64)
        betastarl = np.zeros((T, self.model.num_states), dtype=np.float64)

        return prosodic_hlm_messages_backwards_log(self.cumulative_likelihoods, self.cumulative_prosody_likelihoods, aDl, log_trans_matrix, pi_0, trunc, betal, betastarl)

    def sample_forwards(self, betal, betastarl):
        T = self.T
        aD = np.exp(self.aDl)
        self._letter_stateseq[:] = -1
        stateseq, stateseq_norep, durations_censored = prosodic_hlm_sample_forwards_log(
            self.likelihood_block_word, self.prosody_likelihood_block_word, self.trans_matrix, self.pi_0, self.aDl, self.model.word_list,
            betal, betastarl,
            np.empty(T, dtype=np.int32),[], [])

        self._stateseq = stateseq
        self._stateseq_norep = stateseq_norep
        self._durations_censored = durations_censored
        return self.stateseq, self.stateseq_norep, self.durations_censored

    def clear_caches(self):
        self._apBl = None
        super(WeakLimitProsodicHDPHLMStatesPython, self).clear_caches()

class WeakLimitProsodicHDPHLMStates(WeakLimitProsodicHDPHLMStatesPython, WeakLimitHDPHLMStates):
    pass

def prosodic_hlm_messages_forwards_log(apBl, aDl, palphal):
    T = palphal.shape[0]
    palphal[:] = -np.inf

    if T <= 0:
        return palphal

    apBl_0 = np.cumsum(np.concatenate(([0], apBl[:, 0])))
    palphal[:] = apBl_0[:-1] + apBl[:, 1] + aDl
    return palphal

def prosodic_hlm_messages_backwards_log(cumulative_likelihoods_func, cumulative_prosody_likelihoods_func, aDl, log_trans_matrix, pi_0, trunc, betal, betastarl):
    T = betal.shape[0]

    for t in range(T-1, -1, -1):
        betastarl[t] = np.logaddexp.reduce(
            betal[t:t+trunc] + cumulative_likelihoods_func(t, t+trunc) + cumulative_prosody_likelihoods_func(t, t+trunc) + aDl[:min(trunc, T-t)],
            axis=0
        )
        betal[t-1] = np.logaddexp.reduce(betastarl[t] + log_trans_matrix, axis=1)
    betal[-1] = 0.0
    normalizerl = np.logaddexp.reduce(betastarl[0] + np.log(pi_0))

    return betal, betastarl, normalizerl


def prosodic_hlm_sample_forwards_log(likelihood_block_word_func, prosody_likelihood_block_word_func, trans_matrix, pi_0, aDl, word_list, betal, betastarl, stateseq, stateseq_norep, durations_censored):
    stateseq[:] = -1
    T = betal.shape[0]
    t = 0
    aD = np.exp(aDl)
    nextstate_unsmoothed = pi_0
    while t < T:
        logdomain = betastarl[t] - betastarl[t].max()
        nextstate_dist = np.exp(logdomain) * nextstate_unsmoothed

        state = sample_discrete(nextstate_dist)

        durprob = np.random.random()
        cache_mess_term = np.exp(likelihood_block_word_func(t, T, word_list[state]) + prosody_likelihood_block_word_func(t, T, state) + betal[t:T, state] - betastarl[t, state])

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
        assert dur >= len(word_list[state])
        stateseq[t:t+dur] = state
        nextstate_unsmoothed = trans_matrix[state]
        t += dur

        stateseq_norep.append(state)
        durations_censored.append(dur)
    stateseq_norep = np.array(stateseq_norep, dtype=np.int32)
    durations_censored = np.array(durations_censored, dtype=np.int32)
    return stateseq, stateseq_norep, durations_censored
