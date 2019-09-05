import numpy as np
from pyhsmm.util.general import list_split

from pyhlm.internals import prosodic_hlm_states

from pyhlm.model import WeakLimitHDPHLMPython

class WeakLimitProsodicHDPHLMPython(WeakLimitHDPHLMPython):
    _states_class = prosodic_hlm_states.WeakLimitProsodicHDPHLMStatesPython

    def __init__(self, num_states, alpha, gamma, init_state_concentration, letter_hsmm, prosody_distns, dur_distns, length_distn):
        self._prosody_distns = prosody_distns
        super(WeakLimitProsodicHDPHLMPython, self).__init__(num_states, alpha, gamma, init_state_concentration, letter_hsmm, dur_distns, length_distn)

    @property
    def prosody_distns(self):
        return self._prosody_distns

    def add_data(self, data, prosody_data, **kwargs):
        self.states_list.append(self._states_class(self, data=data, prosody_data=prosody_data, **kwargs))

    def resample_model(self, num_procs=0):
        self.letter_hsmm.states_list = []
        [state.add_word_datas(generate=False) for state in self.states_list]
        self.letter_hsmm.resample_states(num_procs=num_procs)
        [letter_state.reflect_letter_stateseq() for letter_state in self.letter_hsmm.states_list]
        self.resample_words(num_procs=num_procs)
        self.letter_hsmm.resample_parameters_by_sampled_words(self.word_list)
        self.resample_length_distn()
        self.resample_dur_distns()
        self.resample_prosody_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self.resample_states(num_procs=num_procs)
        self._clear_caches()

    def resample_prosody_distns(self):
        F_prosody = [[], []]
        for s in self.states_list:
            F_t = [np.arange(len(s.data))]
            F_t.append(s.durations_censored.cumsum()-1)
            F_t[0] = np.delete(F_t[0], F_t[1])
            F_prosody[0].append(s.prosody_data[F_t[0]])
            F_prosody[1].append(s.prosody_data[F_t[1]])
        self.prosody_distns[0].resample(np.concatenate(F_prosody[0], axis=0))
        self.prosody_distns[1].resample(np.concatenate(F_prosody[1], axis=0))

    def _joblib_resample_states(self,states_list, num_procs):
        from joblib import Parallel, delayed
        from . import parallel

        # warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = list_split(
                    [self._get_prosody_joblib_pair(s) for s in states_list],
                    num_procs)

            parallel.model = self
            parallel.args = joblib_args

            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_prosodic_sampled_stateseq_norep_and_durations_censored)(idx)
                            for idx in range(len(joblib_args)))

            for s, (stateseq, stateseq_norep, durations_censored, log_likelihood) in zip(
                    [s for grp in list_split(states_list,num_procs) for s in grp],
                    [seq for grp in raw_stateseqs for seq in grp]):
                s.stateseq, s._stateseq_norep, s._durations_censored, s._normalizer = stateseq, stateseq_norep, durations_censored, log_likelihood

    def _get_prosody_joblib_pair(self,states_obj):
        return (states_obj.data, states_obj.prosody_data, states_obj._kwargs)

class WeakLimitProsodicHDPHLM(WeakLimitProsodicHDPHLMPython):
    _states_class = prosodic_hlm_states.WeakLimitProsodicHDPHLMStates
