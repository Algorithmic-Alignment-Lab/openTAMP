""" This file defines the sample class. """
import numpy as np

from opentamp.policy_hooks.utils.policy_solver_utils import ACTION_ENUM


class Sample(object):
    """
    Class that handles the representation of a trajectory and stores a
    single trajectory.
    Note: must be serializable for easy saving, no C++ references!
    """
    def __init__(self, agent, T=None):
        self.agent = agent
        self.T = agent.T if T is None else T
        self.step = 0
        self.task_end = False
        self._data = {}
        self.reinit()
        self.draw = True
        self.opt_wt = None

    def reinit(self):
        self.dX = self.agent.dX
        self.dU = self.agent.dU
        self.dO = self.agent.dO
        self.dM = self.agent.dM
        self.dPrim = self.agent.dPrim
        self.dPrimOut = self.agent.dPrimOut
        self.dContOut = self.agent.dContOut
        self.dCont = self.agent.dCont
        self.success = 0
        self.opt_suc = 0
        self._postsuc = False
        self.source_label = 'rollout'
        self.wt = 1.
        self.base_x = None
        self.condition = 0

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self.env_state = {}
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)
        self._prim_out = np.empty((self.T, self.dPrimOut))
        self._prim_out.fill(np.nan)
        self._cont_out = np.empty((self.T, self.dContOut))
        self._cont_out.fill(np.nan)
        self._prim_obs = np.empty((self.T, self.dPrim))
        self._prim_obs.fill(np.nan)
        self._cont_obs = np.empty((self.T, self.dCont))
        self._cont_obs.fill(np.nan)
        self._meta = np.empty(self.dM)
        self._meta.fill(np.nan)
        self._ref_U = np.zeros((self.T, self.dU), dtype='float32')
        self._ref_X = np.zeros((self.T, self.agent.symbolic_bound), dtype='float32')

        self.task_cost = np.nan
        self.task_start = False
        self.removable = True
        self.use_ts = np.ones(self.T)
        self.prim_use_ts = np.ones(self.T)
        self.opt_strength = 0.

    def set(self, sensor_name, sensor_data, t=None):
        """ Set trajectory data for a particular sensor. """
        if t is None:
            self._data[sensor_name] = sensor_data
            self._X.fill(np.nan)  # Invalidate existing X.
            self._obs.fill(np.nan)  # Invalidate existing obs.
            self._prim_obs.fill(np.nan)  # Invalidate existing obs.
            self._prim_out.fill(np.nan)  # Invalidate existing out.
            self._cont_out.fill(np.nan)  # Invalidate existing out.
            self._cont_obs.fill(np.nan)  # Invalidate existing obs.
            self._meta.fill(np.nan)  # Invalidate existing meta data.
        else:
            if sensor_name not in self._data:
                self._data[sensor_name] = \
                        np.empty((self.T,) + sensor_data.shape)
                self._data[sensor_name].fill(np.nan)
            self._data[sensor_name][t, :] = sensor_data
            self._X[t, :].fill(np.nan)
            self._obs[t, :].fill(np.nan)
            self._prim_obs[t, :].fill(np.nan)
            self._prim_out[t, :].fill(np.nan)
            self._cont_obs[t, :].fill(np.nan)
            self._cont_out[t, :].fill(np.nan)

    def get(self, sensor_name, t=None):
        """ Get trajectory data for a particular sensor. """
        return (self._data[sensor_name] if t is None
                else self._data[sensor_name][t, :])

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        X = self._X if t is None else self._X[t, :]
        if np.any(np.isnan(X)):
            for data_type in self._data:
                if data_type not in self.agent.x_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_x(X, data, data_types=[data_type])
        return X.copy()

    def set_X(self, X, t=None):
        for data_type in self.agent._x_data_idx:
            self.set(data_type, X[self.agent._x_data_idx[data_type]], t=t)

    def set_obs(self, obs, t=None):
        for data_type in self.agent._obs_data_idx:
            self.set(data_type, obs[self.agent._obs_data_idx[data_type]], t=t)

    def set_prim_obs(self, prim_obs, t=None):
        for data_type in self.agent._prim_obs_data_idx:
            self.set(data_type, prim_obs[self.agent._prim_obs_data_idx[data_type]], t=t)

    def get_U(self, t=None):
        """ Get the action. """
        # return self._data[ACTION] if t is None else self._data[ACTION][t, :]
        return self._data[ACTION_ENUM] if t is None else self._data[ACTION_ENUM][t, :]

    def get_obs(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._obs if t is None else self._obs[t, :]
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.agent.obs_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])

                if np.any(np.isnan(data)):
                    #breakpoint()
                    print("Nans in ctrl obs for data_type", data_type)

                self.agent.pack_data_obs(obs, data, data_types=[data_type])
        return obs.copy()

    def get_prim_obs(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._prim_obs if t is None else self._prim_obs[t, :]
        # breakpoint()
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.agent.prim_obs_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])

                if np.any(np.isnan(data)):
                    #breakpoint()
                    print("Nans in prim obs for data_type", data_type)
                    
                self.agent.pack_data_prim_obs(obs, data, data_types=[data_type])
        if np.any(np.isnan(obs)):
            #breakpoint()
            print("Still have nans in obs, idxs are", self.agent._prim_obs_idx, self.agent.dPrim)
        return obs.copy()

    def get_prim_out(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        out = self._prim_out if t is None else self._prim_out[t, :]
        if np.any(np.isnan(out)):
            for data_type in self._data:
                if data_type not in self.agent.prim_out_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                # breakpoint()
                self.agent.pack_data_prim_out(out, data, data_types=[data_type])
        return out.copy()

    def get_cont_obs(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._cont_obs if t is None else self._cont_obs[t, :]
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.agent.cont_obs_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_cont_obs(obs, data, data_types=[data_type])
        return obs.copy()

    def get_cont_out(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        out = self._cont_out if t is None else self._cont_out[t, :]
        if np.any(np.isnan(out)):
            for data_type in self._data:
                if data_type not in self.agent.cont_out_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_cont_out(out, data, data_types=[data_type])
        return out.copy()

    def get_meta(self):
        """ Get the meta data. Put it together if not precomputed. """
        meta = self._meta
        if np.any(np.isnan(meta)):
            for data_type in self._data:
                if data_type not in self.agent.meta_data_types:
                    continue
                data = self._data[data_type]
                self.agent.pack_data_meta(meta, data, data_types=[data_type])
        return meta

    def set_ref_X(self, X):
        self._ref_X[:,:] = X[:, :]

    def set_ref_U(self, U):
        self._ref_U[:, :] = U[:, :]

    def get_ref_X(self, t=-1):
        if t > 0:
            return self._ref_X[t].copy()
        return self._ref_X.copy()

    def get_ref_U(self, t=-1):
        if t > 0:
            return self._ref_U[t].copy()
        return self._ref_U.copy()

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('agent')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__['agent'] = None
