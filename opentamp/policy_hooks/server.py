import dill as pickle
from datetime import datetime
import numpy as np
import os, psutil
import pprint
import queue
import random
import sys
import time
import torch
import imageio

from PIL import Image
import pybullet as P

from opentamp.software_constants import *
from opentamp.core.internal_repr.plan import Plan
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.save_video import save_video
from opentamp.policy_hooks.search_node import *


LOG_DIR = 'experiment_logs/'

class Server(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        self._hyperparams = hyperparams
        self.config = hyperparams
        self.group_id = hyperparams['group_id']
        self.debug = hyperparams['debug']
        self.plan_only = hyperparams['plan_only']

        self.start_t = hyperparams['start_t']
        self.seed = int((1e2*time.time()) % 1000.)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.render = hyperparams.get('load_render', False)
        self.weight_dir = self.config['weight_dir']
        self.exp_id = self.weight_dir.split('/')[-1]
        label = self.config['label_server']

        n_gpu = hyperparams['n_gpu']
        if n_gpu == 0:
            gpus = -1
        elif n_gpu == 1:
            gpus = 0
        else:
            #gpus = str(list(range(1, n_gpu+1)))[1:-1]
            #gpus = str(list(range(0, n_gpu)))[1:-1]
            gpus = np.random.choice(range(n_gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = "{0}".format(gpus)
        #os.environ['CUDA_VISIBLE_DEVICES'] = ""

        self.solver = hyperparams['mp_solver_type'](hyperparams)
        self.init_policy_opt(hyperparams)
        hyperparams['agent']['master_config'] = hyperparams
        try:
            P.disconnect()
        except:
            print('No need to disconnect pybullet')
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.process_id = '{0}_{1}'.format(self.id, self.group_id)
        self.agent.solver = self.solver
        self.map_cont_discr_tasks()
        self.prob = self.agent.prob
        self.solver.agent = self.agent
        
        if self.render:
            self.cur_vid_id = 0
            if not os.path.isdir(LOG_DIR+hyperparams['weight_dir']+'/videos'):
                try:
                    os.makedirs(LOG_DIR+hyperparams['weight_dir']+'/videos')
                except:
                    pass

            self.video_dir = LOG_DIR+hyperparams['weight_dir']+'/videos/'

        self.task_queue = hyperparams['task_queue']
        self.motion_queue = hyperparams['motion_queue']
        self.rollout_queue = hyperparams['rollout_queue']
        self.ll_queue = hyperparams['ll_queue']
        self.hl_queue = hyperparams['hl_queue']
        self.cont_queue = hyperparams['cont_queue']

        self.label_type = 'base'
        self._n_plans = 0
        n_plans = hyperparams['policy_opt']['buffer_sizes']['n_plans']
        self._last_weight_read = 0.

        self.permute_hl = hyperparams['permute_hl'] > 0
        self.neg_ratio = hyperparams['perc_negative']
        self.use_neg = self.neg_ratio > 0
        self.opt_ratio = hyperparams['perc_optimal']
        self.dagger_ratio = hyperparams['perc_dagger']
        self.rollout_ratio = hyperparams['perc_rollout']
        self.verbose = hyperparams['verbose']
        self.backup = hyperparams['backup']
        self.end2end = hyperparams['end_to_end_prob']
        self.task_list = self.agent.task_list
        self.stopped = False
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[], 'tasks':[], 'use_mask':[]}
        self.last_log_t = time.time()

        self.current_id = 0
        self.cur_step = 0
        self.adj_eta = False
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.explore_wt = hyperparams['explore_wt']
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        # self.agent.plans, self.agent.openrave_bodies, self.agent.env = self.agent.prob.get_plans(use_tf=True)
        self.dagger_window = hyperparams['dagger_window']
        self.rollout_opt = hyperparams['rollout_opt']
        task_plans = list(self.agent.plans.values())
        for plan in task_plans:
            self._setup_plan(plan)

        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.ff_data_file = LOG_DIR+hyperparams['weight_dir']+'/ff_samples_{0}_{1}.pkl'
        self.log_file = LOG_DIR + hyperparams['weight_dir'] + '/rollout_logs/{0}_log.txt'.format(self.id)
        self.verbose_log_file = LOG_DIR + hyperparams['weight_dir'] + '/rollout_logs/{0}_verbose.txt'.format(self.id)
        self.n_plans = 0
        self.n_failed = 0


    def _setup_plan(self, plan):
        plan.state_inds = self.agent.state_inds
        plan.action_inds = self.agent.action_inds
        plan.dX = self.agent.dX
        plan.dU = self.agent.dU
        plan.symbolic_bound = self.agent.symbolic_bound
        plan.target_dim = self.agent.target_dim
        plan.target_inds = self.agent.target_inds
        for param in plan.params.values():
            for attr in param.attrs:
                if (param.name, attr) not in plan.state_inds:
                    if type(getattr(param, attr)) is not np.ndarray: continue
                    val = getattr(param, attr)[:,0]
                    if np.any(np.isnan(val)):
                        getattr(param, attr)[:] = 0.
                    else:
                        getattr(param, attr)[:,:] = val.reshape((-1,1))


    
    def map_cont_discr_tasks(self):
        self.task_types = []
        self.discrete_opts = []
        self.continuous_opts = []
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for key, val in opts.items():
            if hasattr(val, '__len__'):
                self.task_types.append('discrete')
                self.discrete_opts.append(key)
            else:
                self.task_types.append('continuous')
                self.continuous_opts.append(key)

    
    def init_policy_opt(self, hyperparams):
        config = hyperparams['policy_opt']
        opt_cls = config['type']
        config['gpu_id'] = np.random.randint(0,1)
        config['use_gpu'] = torch.cuda.is_available()
        config['weight_dir'] = hyperparams['weight_dir']
        self.policy_opt = opt_cls(config)
        self.weights_to_store = {}


    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def spawn_problem(self, x0=None, targets=None):
        if x0 is None or targets is None:
            x0, targets = self.new_problem()

        initial, goal = self.agent.get_hl_info(x0, targets)
        problem = self._hyperparams['problem']
        domain = self._hyperparams['domain']
        problem.goal = goal
        abs_prob = self.agent.hl_solver.translate_problem(problem, goal=goal, initial=initial)
        ref_x0 = self.agent.clip_state(x0)
        for pname, attr in self.agent.state_inds:
            p = problem.init_state.params[pname]
            if p.is_symbol(): continue

            inds = self.agent.state_inds[pname, attr]
            getattr(p, attr)[:,0] = ref_x0[inds]

        for targ, attr in self.agent.target_inds:
            if targ in problem.init_state.params:
                p = problem.init_state.params[targ]
                inds = self.agent.target_inds[targ, attr]
                getattr(p, attr)[:,0] = targets[inds].copy()

        prefix = []
        hlnode = HLSearchNode(abs_prob,
                             domain,
                             problem,
                             priority=0,
                             prefix=prefix,
                             llnode=None,
                             x0=x0,
                             targets=targets,
                             label=self.id,
                             info=self.agent.get_hist_info())
        return hlnode


    def new_problem(self):
        x0, targets = self.agent.get_random_initial_state_vec(self.config, 
                                                              self.agent._eval_mode, 
                                                              self.agent.dX, 
                                                              self.agent.state_inds, 
                                                              1,)
        x0, targets = x0[0], targets[0]
        # target_vec = np.zeros(self.agent.target_dim)
        # for (tname, attr), inds in self.agent.target_inds.items():
        #     if attr != 'value': continue
        #     target_vec[inds] = targets[tname]

        return x0, None  # removing targets


    def update(self, obs, mu, prc, wt, task, label, acts=[], ref_acts=[], terminal=[], aux=[], primobs=[], x=[]):
        primobs = [] if task in ['primitive', 'cont', 'label'] or self.end2end == 0 else primobs
        data = (obs, mu, prc, wt, aux, primobs, x, task, label)
        if task == 'primitive':
            q = self.hl_queue

        elif task == 'cont':
            q = self.cont_queue

        elif task == 'label':
            q = self.label_queue

        elif task in self.ll_queue:
            q = self.ll_queue[task]

        elif task in self.task_list:
            q = self.ll_queue['control']

        else:
            raise ValueError('No queue found for {}'.format(task))

        self.push_queue(data, q)


    def policy_call(self, obs, t, noise, task, opt_s=None):
        scope = task if task in self.policy_opt.ctrl_scopes else 'control'

        dO, dU = self.policy_opt._select_dims(scope)
        if noise is None: noise = np.zeros(dU)

        if opt_s is not None:
            if scope == 'primitive':
                return opt_s.get_prim_out(t) + noise

            elif scope == 'cont':
                return opt_s.get_cont_out(t) + noise

            else:
                return opt_s.get_U(t) + noise

        policy = self.policy_opt.get_policy(scope)
        assert policy.is_initialized()

        return policy.act(None, obs, noise)


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None):
        if self.adj_eta: eta *= self.agent.eta_scale
        # distrs = self.policy_opt.task_distr(prim_obs, bounds=self.policy_opt._primBounds, eta=eta)
        distrs = self.policy_opt.task_distr(prim_obs, eta=eta)
        if not soft: return distrs

        out = []
        opts = self.agent.prob.get_prim_choices(self.task_list)
        enums = list(opts.keys())
        for ind, d in enumerate(distrs):
            enum = enums[ind]
            if not np.isscalar(opts[enum]):
                if np.any(np.isnan(d)) or not np.sum(d):
                    d = np.ones(len(d)) / len(d)

                p = d / np.sum(d)
                ind = np.random.choice(list(range(len(d))), p=p)
                d[ind] += 1e2
                d /= np.sum(d)

            out.append(d)

        return out


    def store_weights(self, msg):
        self.weights_to_store[msg.scope] = msg.data


    def update_weights(self):
        scopes = list(self.weights_to_store.keys())
        for scope in scopes:
            save = self.id.endswith('0')
            data = self.weights_to_store[scope]
            self.weights_to_store[scope] = None
            if data is not None:
                self.policy_opt.deserialize_weights(data, save=save)


    def pop_queue(self, q):
        try:
            node = q.get_nowait()
        except queue.Empty:
            node = None

        return node


    def push_queue(self, prob, q):
        if q.full():
            try:
                node = q.get_nowait()
            except queue.Empty:
                node = None

            if node is not None and \
               hasattr(node, 'heuristic') and \
               node.heuristic() < prob.heuristic():
                prob = node

        try:
            q.put_nowait(prob)
        except queue.Full:
            pass


    def set_policies(self):
        inter = 120
        if self.policy_opt.share_buffers and time.time() - self._last_weight_read > inter:
            self.policy_opt.read_shared_weights()
            self._last_weight_read = time.time()

        rollout_policies = {}
        for task in self.agent.task_list:
            rollout_policies[task] = self.policy_opt.get_policy(task)

        rollout_policies['cont'] = self.policy_opt.get_policy('cont')
        
        self.agent.policies = rollout_policies


    def run_hl_update(self, label=None):
        ### Look for saved successful HL rollout paths and send them to update the HL options policy
        ref_paths = []
        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)
            ref_paths.append(path)

        self.agent.clear_task_paths()
        if label is not None:
            for s in path_samples:
                s.source_label = label
        
        self.update_primitive(path_samples)
        if self._hyperparams.get('save_expert', False):
            self.update_expert_demos(ref_paths)


    def run(self):
        raise NotImplementedError()


    def update_policy(self, optimal_samples, label='optimal', inv_cov=None, task=None):
        dU, dO = self.agent.dU, self.agent.dO
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dU))
        tgt_prc, tgt_wt = np.zeros((0, dU, dU)), np.zeros((0,))
        x_data = np.zeros((0, self.agent.dX))
        prim_obs_data = np.zeros((0, self.agent.dPrim))

        for sample in optimal_samples:
            prc = np.zeros((1, sample.T, dU, dU))
            wt = np.zeros((sample.T,))
            for t in range(sample.T):
                if inv_cov is None:
                    traj = self.new_traj_distr[m]
                    prc[:, t, :, :] = np.tile(traj.inv_pol_covar[0, :, :], [1, 1, 1])
                else:
                    prc[:, t, :, :] = np.tile(inv_cov, [1, 1, 1])

                wt[t] = sample.use_ts[t] # self._hyperparams['opt_wt'] * sample.use_ts[t]

            tgt_mu = np.concatenate((tgt_mu, sample.get_U()), axis=0)
            obs_data = np.concatenate((obs_data, sample.get_obs()), axis=0)
            prim_obs_data = np.concatenate((prim_obs_data, sample.get_prim_obs()), axis=0)
            x_data = np.concatenate((x_data, sample.get_X()), axis=0)
            tgt_wt = np.concatenate((tgt_wt, wt), axis=0)
            tgt_prc = np.concatenate((tgt_prc, prc.reshape(-1, dU, dU)), axis=0)

        if task is None:
            task = self.task

        # breakpoint()

        if len(tgt_mu):
            print('Sending update to policy')
            # breakpoint()
            self.update(obs_data, 
                        tgt_mu, 
                        tgt_prc, 
                        tgt_wt, 
                        task, 
                        'optimal', 
                        primobs=prim_obs_data, 
                        x=x_data,)
        else:
            print('WARNING: Update called with no data.')


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(self.agent.discrete_opts)
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, len(self.discrete_opts))), np.zeros((0))
        tgt_aux = np.zeros((0))
        tgt_x = np.zeros((0, self.agent.dX))

        #if len(samples):
        #    lab = samples[0].source_label
        #    lab = 'n_plans' if lab == 'optimal' else 'n_rollout'
        #    if lab in self.policy_opt.buf_sizes:
        #        with self.policy_opt.buf_sizes[lab].get_lock():
        #            self.policy_opt.buf_sizes[lab].value += 1
        #        samples[0].source_label = ''

        for ind, sample in enumerate(samples):
            mu = sample.get_prim_out()
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_x = np.concatenate((tgt_x, sample.get_X()))
            st, et = 0, sample.T # st, et = sample.step * sample.T, (sample.step + 1) * sample.T
            #aux = np.ones(sample.T)
            #if sample.task_start: aux[0] = 0.
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            wt = np.array([sample.prim_use_ts[t] * self.prim_decay**t for t in range(sample.T)])
            if sample.task_start and ind > 0 and sample.opt_strength > 0.999: wt[0] = self.prim_first_wt
            if sample.opt_strength < 1-1e-3: wt[:] *= self.explore_wt
            wt[:] *= sample.wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            if np.any(np.isnan(obs)):
                print("NAN IN OBS PRIM:", obs, sample.task, 'SAMPLE')
            obs_data = np.concatenate((obs_data, obs))
            prc = np.concatenate([self.agent.get_mask(sample, enum) for enum in self.discrete_opts], axis=-1)
            if not self.config['hl_mask']:
                prc[:] = 1.
            # prc = np.ones((sample.T,dP))
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            print('Sending update to primitive net')
            self.update(obs_data, 
                        tgt_mu, 
                        tgt_prc, 
                        tgt_wt, 
                        'primitive', 
                        'optimal', 
                        aux=tgt_aux, 
                        x=tgt_x)


    def update_cont_network(self, samples):
        dP, dO = self.agent.dContOut, self.agent.dCont
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        tgt_aux = np.zeros((0))
        tgt_x = np.zeros((0, self.agent.dX))

        for ind, sample in enumerate(samples):
            mu = sample.get_cont_out()
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_x = np.concatenate((tgt_x, sample.get_X()))
            st, et = 0, sample.T # st, et = sample.step * sample.T, (sample.step + 1) * sample.T
            #aux = np.ones(sample.T)
            #if sample.task_start: aux[0] = 0.
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            wt = np.ones(sample.T)
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_cont_obs()
            if np.any(np.isnan(obs)):
                print("NAN IN OBS CONT:", obs, sample.task, 'SAMPLE')
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dP), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))

        # breakpoint()

        if len(tgt_mu):
            print('Sending update to cont net')
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'cont', 'optimal', aux=tgt_aux, x=tgt_x)


    def get_path_data(self, path, n_fixed=0, verbose=False):
        data = []
        for sample in path:
            X = [{(pname, attr): sample.get_X(t=t)[self.agent.state_inds[pname, attr]].round(3) for pname, attr in self.agent.state_inds if self.agent.state_inds[pname, attr][-1] < self.agent.symbolic_bound} for t in range(sample.T)]
            if hasattr(sample, 'col_ts'):
                U = [{(pname, attr): (sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4), sample.col_ts[t]) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            else:
                U = [{(pname, attr): sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            info = {'X': X, 'task': sample.task, 'time_from_start': time.time() - self.start_t, 'value': 1.-sample.task_cost, 'fixed_samples': n_fixed, 'root_state': self.agent.x0[0], 'opt_strength': sample.opt_strength if hasattr(sample, 'opt_strength') else 'N/A'}
            if verbose:
                info['obs'] = sample.get_obs().round(3)
                # info['prim_obs'] = sample.get_prim_obs().round(3)
                info['targets'] = {tname: sample.targets[self.agent.target_inds[tname, attr]] for tname, attr in self.agent.target_inds if attr == 'value'}
                info['opt_success'] = sample.opt_suc
                # info['tasks'] = sample.get(FACTOREDTASK_ENUM)
                #info['goal_pose'] = sample.get(END_POSE_ENUM)
                info['actions'] = sample.get(ACTION_ENUM)
                info['end_state'] = sample.end_state
                info['plan_fail_rate'] = self.n_failed / self.n_plans if self.n_plans > 0 else 0.
                info['source'] = sample.source_label
                # info['prim_obs'] = sample.get_prim_obs().round(4)
                info['memory'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                info['last_weight_read'] = self._last_weight_read 

            data.append(info)
        return data


    def log_path(self, path, n_fixed=0):
        if self.log_file is None: return
        with open(self.log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed)
            pp_info = pprint.pformat(info, depth=120, width=120)
            f.write(pp_info)
            f.write('\n')

        with open(self.verbose_log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed, True)
            pp_info = pprint.pformat(info, depth=120, width=120)
            f.write(pp_info)
            f.write('\n')


    def save_image(self, rollout=None, success=None, ts=0, render=True, x=None):
        if not self.render: return
        suc_flag = ''
        if success is not None:
            suc_flag = 'succeeded' if success else 'failed'
        fname = '/home/michaelmcdonald/Dropbox/videos/{0}_{1}_{2}.png'.format(self.id, self.cur_vid_id, suc_flag)
        self.cur_vid_id += 1
        if rollout is not None: self.agent.target_vecs[0][:] = rollout.targets
        if render:
            if x is None:
                x = rollout.get_X(t=ts)
            im = self.agent.get_image(x)
        else:
            im = rollout.get(IM_ENUM, t=ts).reshape((self.agent.image_height, self.agent.image_width, 3))
            im = (128 * im) + 128
            im = im.astype(np.uint8)
        im = Image.fromarray(im)
        im.save(fname)


    def _gen_video(self, rollout, st=0, ts=None, annotate=False, tdelta=1):
        if not self.render: return None
        old_h = self.agent.image_height
        old_w = self.agent.image_width
        self.agent.image_height = 256
        self.agent.image_width = 256
        cam_ids = self.config.get('visual_cameras', [self.agent.camera_id])
        buf = []
        for step in rollout:
            if not step.draw: continue
            old_vec = self.agent.target_vecs[0]
            self.agent.target_vecs[0] = step.targets
            ts = (st, step.T) if ts is None else ts
            ts_range = range(ts[0], ts[1], tdelta)
            st = 0

            for t in ts_range:
                if t >= step.T: break
                ims = []
                for ind, cam_id in enumerate(cam_ids):
                    if annotate and ind == 0:
                        ims.append(self.agent.get_annotated_image(step, t, cam_id=cam_id))
                    else:
                        ims.append(self.agent.get_image(step.get_X(t=t), cam_id=cam_id))
                im = np.concatenate(ims, axis=1)
                buf.append(im)
            self.agent.target_vecs[0] = old_vec
        self.agent.image_height = old_h
        self.agent.image_width = old_w
        return np.array(buf)


    def save_video(self, rollout, success=None, ts=None, lab='', annotate=True, st=0):
        if not self.render: return
        init_t = time.time()
        old_h = self.agent.image_height
        old_w = self.agent.image_width
        self.agent.image_height = 256
        self.agent.image_width = 256
        suc_flag = ''
        cam_ids = self.config.get('visual_cameras', [self.agent.camera_id])
        if success is not None:
            suc_flag = 'success' if success else 'fail'

        fname = self.video_dir + '/{0}_{1}_{2}_{3}{4}_{5}.npy'.format(self.id, 
                                                                      self.group_id, 
                                                                      self.cur_vid_id, 
                                                                      suc_flag, 
                                                                      lab, 
                                                                      str(cam_ids)[1:-1].replace(' ', ''), )
        self.cur_vid_id += 1
        buf = []
        for step in rollout:
            if not step.draw: continue
            old_vec = self.agent.target_vecs[0]
            self.agent.target_vecs[0] = step.targets
            if ts is None: 
                ts_range = range(st, step.T)
            else:
                ts_range = range(ts[0], ts[1])
            st = 0

            for t in ts_range:
                ims = []
                for ind, cam_id in enumerate(cam_ids):
                    if annotate and ind == 0:
                        ims.append(self.agent.get_annotated_image(step, t, cam_id=cam_id))
                    else:
                        ims.append(self.agent.get_image(step.get_X(t=t), cam_id=cam_id))
                im = np.concatenate(ims, axis=1)
                buf.append(im)
            self.agent.target_vecs[0] = old_vec
        init_t = time.time()
        # TODO: hardcoded individual video name for now
        print('Saving video to: ', LOG_DIR+self._hyperparams['weight_dir']+'/videos/'+lab+'.gif')
        imageio.mimsave(LOG_DIR+self._hyperparams['weight_dir']+'/videos/'+lab+'.gif', np.array(buf), duration=0.01)
        save_video(fname, dname=self._hyperparams['descr'], arr=np.array(buf), savepath=self.video_dir)
        self.agent.image_height = old_h
        self.agent.image_width = old_w

