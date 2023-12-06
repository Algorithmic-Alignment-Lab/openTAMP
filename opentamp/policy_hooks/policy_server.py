import pickle
import pprint
import random
import threading
import time
import queue
import numpy as np
import os, psutil

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from opentamp.policy_hooks.fast_dataloader import FastDataLoader
from opentamp.policy_hooks.queued_dataset import QueuedDataset
from opentamp.policy_hooks.utils.file_utils import LOG_DIR
from opentamp.policy_hooks.utils.policy_solver_utils import *


MAX_QUEUE_SIZE = 100
UPDATE_TIME = 60

class PolicyServer(object):
    def __init__(self, hyperparams):
        self.config = hyperparams
        self.group_id = hyperparams['group_id']
        self.task = hyperparams['scope']
        self.task_list = hyperparams['task_list']

        base_log_dir = hyperparams.get('log_dir', LOG_DIR)
        self.weight_dir = base_log_dir+hyperparams['weight_dir']

        self.seed = int((1e2*time.time()) % 1000.)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_gpu = hyperparams['n_gpu']
        if n_gpu == 0:
            gpus = -1
        elif n_gpu == 1:
            gpus = 0
        else:
            gpus = np.random.choice(range(n_gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = "{0}".format(gpus)

        self.start_t = hyperparams['start_t']
        self.config = hyperparams
        self.permute = hyperparams['permute_hl'] > 0

        ratios = hyperparams.get('ratios', {})
        for key in ['negative', 'optimal', 'dagger', 'rollout', 'human']:
            if key not in ratios: ratios[key] = hyperparams['perc_{}'.format(key)]
        for key in ratios: ratios[key] /= np.sum(list(ratios.values()))
        hyperparams['ratios'] = ratios

        hyperparams['policy_opt']['scope'] = self.task
        hyperparams['policy_opt']['gpu_id'] = 0
        hyperparams['policy_opt']['use_gpu'] = 1
        hyperparams['policy_opt']['load_all'] = self.task not in ['cont', 'primitive']
        hyperparams['agent']['master_config'] = hyperparams
        hyperparams['agent']['load_render'] = False

        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.map_cont_discr_tasks()
        self.prim_opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        self.stopped = False
        self.queues = hyperparams['queues']
     
        hyperparams['dPrim'] = len(hyperparams['prim_bounds'])
        hyperparams['dCont'] = len(hyperparams['cont_bounds'])
        hyperparams['policy_opt']['hl_network_params']['output_boundaries'] = self.discr_bounds
        hyperparams['policy_opt']['cont_network_params']['output_boundaries'] = self.cont_bounds
        hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir']
        hyperparams['policy_opt']['dPrimObs'] = self.agent.dPrim
        hyperparams['policy_opt']['dContObs'] = self.agent.dCont
        hyperparams['policy_opt']['dO'] = self.agent.dO

        self.policy_opt = hyperparams['policy_opt']['type'](hyperparams['policy_opt'])
        self.policy_opt.lr_schedule = hyperparams['lr_schedule']
        self.lr_schedule = hyperparams['lr_schedule']

        self._compute_idx()
        self._setup_dataloading()
        self.dataset.policy = self.policy_opt.get_policy(self.task)
        self.dataset.data_buf.policy = self.policy_opt.get_policy(self.task)
        self.policy_opt.data_loader = self.data_gen
        self.debug = hyperparams['debug']
        self.plan_only = hyperparams['plan_only']

        self._setup_log_info()


    def _compute_idx(self):
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        for sensor in self.config['policy_opt']['ll_network_params']['obs_include']:
            dim = self.config['policy_opt']['ll_network_params']['sensor_dims'][sensor]
            if sensor in self.config['policy_opt']['ll_network_params'].get('obs_image_data', []):
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim

        self.prim_x_idx, self.prim_img_idx, i = [], [], 0
        for sensor in self.config['policy_opt']['hl_network_params']['obs_include']:
            dim = self.config['policy_opt']['hl_network_params']['sensor_dims'][sensor]
            if sensor in self.config['policy_opt']['hl_network_params'].get('obs_image_data', []):
                self.prim_img_idx = self.prim_img_idx + list(range(i, i+dim))
            else:
                self.prim_x_idx = self.prim_x_idx + list(range(i, i+dim))
            i += dim

        self.cont_x_idx, self.cont_img_idx, i = [], [], 0
        for sensor in self.config['policy_opt']['cont_network_params']['obs_include']:
            dim = self.config['policy_opt']['cont_network_params']['sensor_dims'][sensor]
            if sensor in self.config['policy_opt']['cont_network_params'].get('obs_image_data', []):
                self.cont_img_idx = self.cont_img_idx + list(range(i, i+dim))
            else:
                self.cont_x_idx = self.cont_x_idx + list(range(i, i+dim))
            i += dim


    def _setup_dataloading(self):
        x_idx = self.x_idx
        if self.task == 'primitive':
            x_idx = self.prim_x_idx
        elif self.task == 'cont':
            x_idx = self.cont_x_idx

        self.min_buffer = self.config['prim_update_size'] if self.task in ['cont', 'primitive'] else self.config['update_size']
        if self.task == 'label':
            self.min_buffer = 600

        if self.task == 'primitive':
            self.in_queue = self.config['hl_queue']
        elif self.task == 'cont':
            self.in_queue = self.config['cont_queue']
        elif self.task == 'label':
            self.in_queue = self.config['label_queue']
        else:
            self.in_queue = self.config['ll_queue'][self.task]

        self.batch_size = self.config['batch_size']
        if self.task in ['cont', 'primitive', 'label']:
            normalize = False
        else:
            normalize = IM_ENUM not in self.config['obs_include']

        feed_prob = self.config['end_to_end_prob']
        in_inds, out_inds = None, None
        if len(self.continuous_opts):
            in_inds, out_inds = [], []
            opt1 = None
            if END_POSE_ENUM in self.agent._obs_data_idx:
                opt1 = END_POSE_ENUM

            for opt in self.continuous_opts:
                inopt = opt1 if opt1 is not None else opt
                in_inds.append(self.agent._obs_data_idx[inopt])
                out_inds.append(self.agent._cont_out_data_idx[opt])
                
            in_inds = np.concatenate(in_inds, axis=0)
            out_inds = np.concatenate(out_inds, axis=0)

        aug_f = None
        no_im = IM_ENUM not in self.config['prim_obs_include']
        if self.task == 'primitive' and self.config['permute_hl'] > 0 and no_im:
            aug_f = self.agent.permute_hl_data
 
        self.dataset = QueuedDataset(self.task, 
                                     self.in_queue, 
                                     self.batch_size,
                                     x_idx=x_idx,
                                     ratios=self.config['ratios'],
                                     normalize=normalize, 
                                     min_buffer=self.min_buffer, 
                                     aug_f=aug_f, 
                                     feed_prob=feed_prob, 
                                     feed_inds=(in_inds, out_inds), 
                                     feed_map=self.agent.center_cont, 
                                     save_dir=self.weight_dir+'/samples/')
        
        self.data_gen = TorchDataLoader(self.dataset, batch_size=self.batch_size)


    def _setup_log_info(self):
        self.policy_opt_log = LOG_DIR + self.config['weight_dir'] + '/policy_{0}_log.txt'.format(self.task)
        self.policy_info_log = LOG_DIR + self.config['weight_dir'] + '/policy_{0}_info.txt'.format(self.task)
        self.data_file = LOG_DIR + self.config['weight_dir'] + '/{0}_data.pkl'.format(self.task)
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[]}
        self.expert_data_file = LOG_DIR+self.config['weight_dir']+'/'+str(self.task)+'_exp_data.npy'
        self.n_updates = 0
        self.update_t = time.time()
        self.n_data = []
        self.update_queue = []
        self.policy_loss = []
        self.train_losses = {'all': [], 'optimal':[], 'rollout':[], 'aux': []}
        self.val_losses = {'all': [], 'optimal':[], 'rollout':[], 'aux': []}
        self.policy_component_loss = []
        self.log_infos = []
        self.cur_ratio = 1.
        with open(self.policy_opt_log, 'w+') as f:
            f.write('')


    def map_cont_discr_tasks(self):
        self.task_types = []
        self.discrete_opts = []
        self.continuous_opts = []
        self.discr_bounds = []
        self.cont_bounds = []
        cur_discr = 0
        cur_cont = 0
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for key, val in opts.items():
            if hasattr(val, '__len__'):
                self.task_types.append('discrete')
                self.discrete_opts.append(key)
                next_discr = cur_discr + len(val)
                self.discr_bounds.append((cur_discr, next_discr))
                cur_discr = next_discr
            else:
                self.task_types.append('continuous')
                self.continuous_opts.append(key)
                next_cont = cur_cont + int(val)
                self.cont_bounds.append((cur_cont, next_cont))
                cur_cont = next_cont


    def run(self):
        self.iters = 0
        write_freq = 50
        
        while not self.stopped:
            self.iters += 1
            init_t = time.time()
            if self.debug or self.plan_only:
                if self.dataset.should_wait_for_data():
                    break
            else:
                self.dataset.wait_for_data()

            self.dataset.load_data()
            #if self.task == 'primitive': print('\nTime to get update:', time.time() - init_t, '\n')
            self.policy_opt.update(self.task)
            #if self.task == 'primitive': print('\nTime to run update:', time.time() - init_t, '\n')
            self.n_updates += 1
            mu, obs, prc = self.dataset.get_batch()
            if len(mu):
                losses = self.policy_opt.check_validation(mu, obs, prc, task=self.task)
                self.train_losses['all'].append(losses[0])
                self.train_losses['aux'].append(losses)

            mu, obs, prc = self.dataset.get_batch(val=True)
            if len(mu): 
                losses = self.policy_opt.check_validation(mu, obs, prc, task=self.task)
                self.val_losses['all'].append(losses[0])
                self.val_losses['aux'].append(losses)

            if self.lr_schedule == 'adaptive':
                if len(self.train_losses['optimal']) and len(self.val_losses['optimal']) and self.policy_opt.cur_dec > 0:
                    ratio = np.mean(self.val_losses['optimal'][-5:]) / np.mean(self.train_losses['optimal'][-5:])
                    self.cur_ratio = ratio
                    if ratio < 1.2:
                        self.policy_opt.cur_dec *= 0.975
                    elif ratio > 1.7:
                        self.policy_opt.cur_dec *= 1.025
                    self.policy_opt.cur_dec = max(self.policy_opt.cur_dec, 1e-12)
                    self.policy_opt.cur_dec = min(self.policy_opt.cur_dec, 1e-1)

            for lab in ['optimal', 'rollout']:
                mu, obs, prc = self.dataset.get_batch(label=lab, val=True)
                if len(mu): self.val_losses[lab].append(self.policy_opt.check_validation(mu, obs, prc, task=self.task)[0])

            if not self.iters % write_freq or (self.debug or self.plan_only):
                self.policy_opt.write_shared_weights([self.task])
                if len(self.continuous_opts) and self.task not in ['cont', 'primitive', 'label']:
                    self.policy_opt.read_shared_weights(['cont'])
                    self.dataset.feed_in_policy = self.policy_opt.cont_policy

                n_train = self.dataset.get_size()
                n_val = self.dataset.get_size(val=True)
                print('Ran', self.iters, 'updates on', self.task, 'with', n_train, 'train and', n_val, 'val')

            if self.config['save_data']:
                if not self.iters % write_freq and self.task in ['cont', 'primitive']:
                    self.dataset.write_data(n_data=1024)

            if (not self.iters % write_freq or (self.debug or self.plan_only)) and len(self.val_losses['all']):
                with open(self.policy_opt_log, 'a+') as f:
                    info = self.get_log_info()
                    pp_info = pprint.pformat(info, depth=60)
                    f.write(str(pp_info))
                    f.write('\n\n')
            
            if self.debug or self.plan_only:
                break # stop iteration after one loop
            
            #if self.task == 'primitive': print('\nTime to finish update:', time.time() - init_t, '\n')


    def get_log_info(self):
        test_acc, train_acc = -1, -1
        test_component_acc, train_component_acc = -1, -1
        #if self.task == 'primitive':
        #    obs, mu, prc = self.data_gen.get_batch()
        #    train_acc = self.policy_opt.task_acc(obs, mu, prc)
        #    train_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
        #    obs, mu, prc = self.data_gen.get_batch(val=True)
        #    test_acc = self.policy_opt.task_acc(obs, mu, prc)
        #    test_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
        info = {
                'time': time.time() - self.start_t,
                'train_loss': np.mean(self.train_losses['all'][-10:]),
                'train_component_loss': np.mean(self.train_losses['all'][-10:], axis=0),
                'val_loss': np.mean(self.val_losses['all'][-10:]),
                'val_component_loss': np.mean(self.val_losses['all'][-10:], axis=0),
                'scope': self.task,
                'n_updates': self.n_updates,
                'n_data': self.policy_opt.N,
                'torch_iter': self.policy_opt.torch_iter,
                'N': self.policy_opt.N,
                'reg_val': self.policy_opt.cur_dec,
                'loss_ratio': self.cur_ratio,
                }

        info['memory'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        # info['in_queue_size'] = self.in_queue.qsize()

        for key in self.dataset.data_buf.lens:
            info['n_train_{}'.format(key)] = self.dataset.data_buf.lens[key]

        for key in self.policy_opt.buf_sizes:
            if key.find('n_') >= 0:
                 info[key] = self.policy_opt.buf_sizes[key].value

        info['labels'] = list(self.dataset.data_buf.lens.keys())
        if len(self.val_losses['rollout']):
            info['rollout_val_loss'] = np.mean(self.val_losses['rollout'][-10:]),
            info['rollout_val_component_loss'] = np.mean(self.val_losses['rollout'][-10:], axis=0),
        if len(self.val_losses['optimal']):
            info['optimal_val_loss'] = np.mean(self.val_losses['optimal'][-10:]),
            info['optimal_val_component_loss'] = np.mean(self.val_losses['optimal'][-10:], axis=0),
        if len(self.train_losses['optimal']):
            info['optimal_train_loss'] = np.mean(self.train_losses['optimal'][-10:]),
            info['optimal_train_component_loss'] = np.mean(self.train_losses['optimal'][-10:], axis=0),
        if test_acc >= 0:
            info['test_accuracy'] = test_acc
            info['test_component_accuracy'] = test_component_acc
            info['train_accuracy'] = train_acc
            info['train_component_accuracy'] = train_component_acc
        #self.log_infos.append(info)
        return info #self.log_infos


    def update_expert_demos(self, obs, acs, rew=None):
        self.expert_demos['acs'].append(acs)
        self.expert_demos['obs'].append(obs)
        if rew is None:
            rew = np.ones(len(obs))
        self.expert_demos['ep_rets'].append(rew)
        self.expert_demos['rews'].append(rew)
        if self.n_updates % 200:
            np.save(self.expert_data_file, self.expert_demos)

