import argparse
import copy
import ctypes
import os
import os.path
import pickle
import psutil
from queue import PriorityQueue
import random
import signal
import sys
from threading import Thread
import time
import traceback

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Pool, Queue

import numpy as np

from opentamp.policy_hooks.motion_server import MotionServer
from opentamp.policy_hooks.policy_opt import TorchPolicyOpt
from opentamp.policy_hooks.policy_server import PolicyServer
from opentamp.policy_hooks.rollout_server import RolloutServer
from opentamp.policy_hooks.task_server import TaskServer
import opentamp.policy_hooks.hl_retrain as hl_retrain
from opentamp.policy_hooks.utils.load_agent import *
from opentamp.policy_hooks.utils.load_task_definitions import *
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.utils.file_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.utils.file_utils import load_config, check_dirs


def spawn_server(cls, hyperparams, load_at_spawn=False):
    if load_at_spawn:
        new_config, config_mod = load_config(hyperparams['args'])
        new_config.update(hyperparams)
        hyperparams = new_config
        if 'main' not in hyperparams:
            hyperparams['main'] = MultiProcessMain(hyperparams)

        hyperparams['main'].init(hyperparams)
        hyperparams['policy_opt']['share_buffer'] = True
        hyperparams['policy_opt']['buffers'] = hyperparams['buffers']
        hyperparams['policy_opt']['buffer_sizes'] = hyperparams['buffer_sizes']

        if cls is PolicyServer \
           and hyperparams['scope'] is 'cont' \
           and not len(hyperparams['cont_bounds']):
            return

    server = cls(hyperparams)
    server.run()


# This enables multiprocessing for priority queues
class QueueManager(SyncManager):
    pass
QueueManager.register('PriorityQueue', PriorityQueue)


class MultiProcessMain(object):
    def __init__(self, config, load_at_spawn=False):
        self.monitor = True
        self.cpu_use = []
        self.config = config
        if load_at_spawn:
            task_file = config.get('task_map_file', '')
            self.pol_list = ('control',) if not config['args'].split_nets else tuple(get_tasks(task_file).keys())
            # config['main'] = self

        else:
            task_file = config.get('task_map_file', '')
            self.pol_list = ('control',) if not config['args'].split_nets else tuple(get_tasks(task_file).keys())
            new_config, config_mod = load_config(config['args'])
            new_config.update(config)
            setup_dirs(self.config, self.config['args'])
            check_dirs(self.config)
            self.init(new_config)


    def init(self, config):
        self.config = config
        self.config['group_id'] = config.get('group_id', 0)
        if 'id' not in self.config: self.config['id'] = -1
        self.config['start_t'] = time.time()

        prob = config['prob']
        self.task_list = tuple(sorted(list(get_tasks(self.config['task_map_file']).keys())))
        self.config['task_list'] = self.task_list
        self.weight_dir = self.config['weight_dir']

        config['agent'] = load_agent(config)

        # Build a local agent to verify some values, but don't lincur rendering voerhead
        old_render = self.config['agent']['master_config']['load_render']
        self.config['agent']['master_config']['load_render'] = False
        self.agent = self.config['agent']['type'](self.config['agent'])
        self.config['agent']['master_config']['load_render'] = old_render

        self._map_cont_discr_tasks()
        self._set_alg_config()
        self.processes = []


    def _map_cont_discr_tasks(self):
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


    def _set_alg_config(self):
    #     if self.config.get('add_hl_image', False):
    #         primitive_network_model = fp_multi_modal_discr_network

    #     elif self.config.get('conditional', False):
    #         primitive_network_model = tf_cond_classification_network

    #     elif self.config.get('discrete_prim', True):
    #         primitive_network_model = tf_classification_network

    #     else:
    #         primitive_network_model = tf_network

    #     if self.config.get('add_cont_image', False):
    #         cont_network_model = fp_multi_modal_cont_network

    #     elif self.config.get('conditional', False):
    #         cont_network_model = tf_cond_classification_network

    #     elif self.config.get('discrete_prim', True):
    #         cont_network_model = tf_classification_network

    #     else:
    #         cont_network_model = tf_network

    #     if self.config.get('add_image', False):
    #         network_model = fp_cont_network

    #     else:
    #         network_model = tf_network

        sensor_dims = self.config['agent']['sensor_dims']
        obs_image_data = [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM]
        add_image = self.config.get('add_image', False)
        add_hl_image = self.config.get('add_hl_image', False)


        self.config['policy_opt'] = {
            'll_policy': self.config.get('ll_policy', ''),
            'hl_policy': self.config.get('hl_policy', ''),
            'cont_policy': self.config.get('cont_policy', ''),
            'type': self.config.get('policy_opt_cls', TorchPolicyOpt),
            'prim_obs_include': self.config['agent']['prim_obs_include'],
            'prim_out_include': self.config['agent']['prim_out_include'],
            
            'll_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                'out_include': [ACTION_ENUM],
                'obs_image_data': obs_image_data,
                'idx': self.agent._obs_data_idx,
                'sensor_dims': sensor_dims,
                'num_filters': self.config.get('num_filters', [32, 32]),
                'filter_sizes': self.config.get('filter_sizes', [7,5]),
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'n_layers': self.config['n_layers'],
                'dim_hidden': self.config['dim_hidden'],
                'act_fn': self.config.get('act_fn', 'relu'),
                'output_fn': self.config.get('output_fn', None),
                'loss_fn': self.config.get('loss_fn', 'precision_mse'),
                'conv_to_fc': 'fp',
                'normalize': not add_image,
                'build_conv': add_image,
                'use_precision': True,
            },

            'hl_network_params': {
                'obs_include': self.config['agent']['prim_obs_include'],
                'out_include': self.config['agent']['prim_out_include'],
                'obs_image_data': obs_image_data,
                'idx': self.agent._prim_obs_data_idx,
                'sensor_dims': sensor_dims,
                'num_filters': self.config.get('prim_filters', [32, 32]),
                'filter_sizes': self.config.get('prim_filter_sizes', [7,5]),
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'n_layers': self.config['prim_n_layers'],
                'dim_hidden': self.config['prim_dim_hidden'],
                'output_boundaries': self.config['prim_bounds'],
                'types': self.task_types,
                'act_fn': self.config.get('act_fn', 'relu'),
                'output_fn': self.config.get('output_fn', 'log_softmax'),
                'loss_fn': self.config.get('loss_fn', 'nll_loss'),
                'conv_to_fc': 'fp',
                'normalize': False,
                'build_conv': add_hl_image,
                'use_precision': True,
            },
    
            'cont_network_params': {
                'obs_include': self.config['agent']['cont_obs_include'],
                'out_include': self.config['agent']['cont_out_include'],
                'obs_image_data': obs_image_data,
                'idx': self.agent._cont_obs_data_idx,
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': sensor_dims,
                'n_layers': self.config['prim_n_layers'],
                'num_filters': self.config.get('cont_filters', [32, 32]),
                'filter_sizes': self.config.get('cont_filter_sizes', [5, 5]),
                'dim_hidden': self.config['prim_dim_hidden'],
                'output_boundaries': self.config['cont_bounds'],
                'types': self.task_types,
                'act_fn': self.config.get('act_fn', 'relu'),
                'output_fn': self.config.get('output_fn', None),
                'loss_fn': self.config.get('loss_fn', 'precision_mse'),
                'conv_to_fc': 'fp',
                'normalize': False,
                'build_conv': add_hl_image,
                'use_precision': True,
            },

            'lr': self.config['lr'],
            'hllr': self.config['hllr'],

            'weight_decay': self.config['weight_decay'],
            'prim_weight_decay': self.config['prim_weight_decay'],
            'cont_weight_decay': self.config['cont_weight_decay'],

            'update_size': self.config['update_size'],
            'prim_update_size': self.config['prim_update_size'],

            'batch_size': self.config['batch_size'],
            'weights_file_prefix': 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list,
            'gpu_fraction': 0.25,
            'allow_growth': True,
            'split_nets': self.config.get('split_nets', False),
            'dPrimObs': self.agent.dPrim,
            'dContObs': self.agent.dCont,
            'dO': self.agent.dO,
        }


    def allocate_shared_buffers(self, config):
        buffers = {}
        buf_sizes = {}
        power = 26

        for task in self.pol_list:
            buffers[task] = mp.Array(ctypes.c_char, (2**power))
            buf_sizes[task] = mp.Value('i')
            buf_sizes[task].value = 0

        buffers['primitive'] = mp.Array(ctypes.c_char, 20 * (2**power))
        buf_sizes['primitive'] = mp.Value('i')
        buf_sizes['primitive'].value = 0
        buffers['cont'] = mp.Array(ctypes.c_char, 20 * (2**power))
        buf_sizes['cont'] = mp.Value('i')
        buf_sizes['cont'].value = 0

        for key in ['optimal', 'human', 'dagger', 'rollout']:
            buf_sizes['n_plan_{}_failed'.format(key)] = mp.Value('i')
            buf_sizes['n_plan_{}_failed'.format(key)].value = 0
            buf_sizes['n_plan_{}'.format(key)] = mp.Value('i')
            buf_sizes['n_plan_{}'.format(key)].value = 0

        for key in ['n_mcts', 'n_postcond', 'n_precond', 'n_midcond',
                    'n_explore', 'n_rollout', 'n_total', 'n_negative',
                    'n_failed', 'n_data', 'n_ff', 'n_plans']:

            buf_sizes[key] = mp.Value('i')
            buf_sizes[key].value = 0

        config['share_buffer'] = True
        config['buffers'] = buffers
        config['buffer_sizes'] = buf_sizes


    def spawn_servers(self, config):
        self.processes = []
        self.process_info = []
        self.process_configs = {}
        self.threads = []
        self.create_servers(config)


    def start_servers(self):
        n_proc = len(self.processes)
        processes = self.processes
        self.processes = []
        for p_ind in range(n_proc):
            # Guard to prevent passing child processes to child processes,
            # which crashes the code on a mac (and possibly other OSes)

            processes[p_ind].start()

            time.sleep(0.1)

        for t in self.threads:
            t.start()

        self.processes = processes

    def create_server(self, server_cls, hyperparams, process=True):
        if hyperparams.get('seq', False):
            spawn_server(server_cls, hyperparams, True)
            sys.exit(0)

        if process:
            p = Process(target=spawn_server, args=(server_cls, hyperparams, True))
            p.name = str(server_cls) + '_run_training'
            p.daemon = True
            self.processes.append(p)
            server_id = hyperparams['id'] if 'id' in hyperparams else hyperparams['scope']
            self.process_info.append((server_cls, server_id))
            self.process_configs[p.pid] = (server_cls, hyperparams)
            return p
        else:
            t = Thread(target=spawn_server, args=(server_cls, hyperparams))
            t.daemon = True
            self.threads.append(t)
            return t


    def create_pol_servers(self, hyperparams):
        for task in self.pol_list+('primitive', 'cont'):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['scope'] = task
            new_hyperparams['id'] = task
            self.create_server(PolicyServer, new_hyperparams)


    def create_servers(self, hyperparams, start_idx=0):
        if hyperparams.get('seq', False):
            self._create_server(hyperparams, TaskServer, 0)

        self.create_pol_servers(hyperparams)
        hyperparams['view'] = False
        for n in range(hyperparams['num_motion']):
            self._create_server(hyperparams, MotionServer, start_idx+n)

        for n in range(hyperparams['num_task']):
            self._create_server(hyperparams, TaskServer, start_idx+n)

        for n in range(hyperparams['num_rollout']):
            self._create_server(hyperparams, RolloutServer, start_idx+n)

        hyperparams = copy.copy(hyperparams)
        hyperparams['run_hl_test'] = True
        hyperparams['id'] = 'test'
        hyperparams['view'] = hyperparams['view_policy']
        hyperparams['load_render'] = hyperparams['load_render'] or hyperparams['view_policy']
        hyperparams['check_precond'] = False
        self.create_server(RolloutServer, copy.copy(hyperparams))

        hyperparams['id'] = 'moretest'
        hyperparams['view'] = False
        self.create_server(RolloutServer, copy.copy(hyperparams))

        for n in range(hyperparams['num_test']):
            hyperparams['id'] = 'server_test_{}'.format(n)
            hyperparams['view'] = False 
            hyperparams['load_render'] = hyperparams['load_render'] or hyperparams['view_policy']
            hyperparams['check_precond'] = False
            self.create_server(RolloutServer, copy.copy(hyperparams))

        hyperparams['run_hl_test'] = False


    def _create_server(self, hyperparams, cls, idx):
        hyperparams = copy.copy(hyperparams)
        hyperparams['id'] = cls.__name__ + str(idx)
        p = self.create_server(cls, hyperparams)
        return p


    def kill_processes(self):
        for p in self.processes:
            p.terminate()

    def check_processes(self):
        states = []
        for n in range(len(self.processes)):
            p = self.processes[n]
            states.append(p.exitcode)
        return states


    def watch_processes(self, kill_all=False):
        exit = False
        while not exit and len(self.processes):
            for n in range(len(self.processes)):
                p = self.processes[n]
                if not p.is_alive():
                    message = 'Killing All.' if kill_all else 'Restarting Dead Process.'
                    print('\n\nProcess died: ' + str(self.process_info[n]) + ' - ' + message)
                    exit = kill_all
                    if kill_all: break
                    process_config = self.process_configs[p.pid]
                    del self.process_info[n]
                    self.create_server(*process_config)
                    print("Relaunched dead process")
            time.sleep(60)
            self.log_mem_info()

        for p in self.processes:
            if p.is_alive(): p.terminate()


    def terminate_processes(self):
        for p in self.processes:
            if p.is_alive(): p.terminate()

        print("Terminate signal sent to children.")
        sys.exit(0)


    def start(self, kill_all=False):
        #self.check_dirs()
        setup_dirs(self.config, self.config['args'])
        if self.config.get('share_buffer', True):
            self.allocate_shared_buffers(self.config)
            self.allocate_queues(self.config)

        signal.signal(signal.SIGINT, lambda sig, frame: self.terminate_processes())
        self.spawn_servers(self.config)
        self.start_servers()

        if self.monitor:
            self.watch_processes(kill_all)


    # def expand_rollout_servers(self):
    #     if not self.config['expand_process'] or time.time() - self.config['start_t'] < 1200: return
    #     self.cpu_use.append(psutil.cpu_percent(interval=1.))
    #     if np.mean(self.cpu_use[-1:]) < 92.5:
    #         hyp = copy.copy(self.config)
    #         hyp['split_mcts_alg'] = True
    #         hyp['run_alg_updates'] = False
    #         hyp['run_mcts_rollouts'] = True
    #         hyp['run_hl_test'] = False
    #         print(('Starting rollout server {0}'.format(self.cur_n_rollout)))
    #         p = self._create_rollout_server(hyp, idx=self.cur_n_rollout)
    #         try:
    #             p.start()
    #         except Exception as e:
    #             print(e)
    #             print('Failed to expand rollout servers')
    #         time.sleep(1.)


    def log_mem_info(self):
        '''
        Get list of running process sorted by Memory Usage
        '''
        listOfProcObjects = []
        # Iterate over the list
        for proc in psutil.process_iter():
            try:
                # Fetch process details as dict
                pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
                pinfo['vms'] = proc.memory_info().vms / (1024 * 1024)
                # Append dict to list
                listOfProcObjects.append(pinfo);
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Sort list of dict by key vms i.e. memory usage
        listOfProcObjects = sorted(listOfProcObjects, key=lambda procObj: procObj['vms'], reverse=True)

        return listOfProcObjects


    def allocate_queues(self, config):
        self.queue_manager = QueueManager()
        self.queue_manager.start()

        queue_size = 50
        train_queue_size = 20

        queues = {}
        config['hl_queue'] = Queue(maxsize=train_queue_size)
        config['ll_queue'] = {} 
        for task in self.pol_list:
            config['ll_queue'][task] = Queue(maxsize=train_queue_size)
        config['cont_queue'] = Queue(maxsize=train_queue_size)

        config['motion_queue'] = self.queue_manager.PriorityQueue(maxsize=queue_size)
        config['task_queue'] = self.queue_manager.PriorityQueue(maxsize=queue_size)
        config['rollout_queue'] = self.queue_manager.PriorityQueue(maxsize=queue_size)

        #for task in self.pol_list+('primitive',):
        #    queues['{0}_pol'.format(task)] = Queue(50)
        config['queues'] = queues
        return queues


    def hl_only_retrain(self):
        hyperparams = self.config
        hyperparams['id'] = 'test'
        hyperparams['scope'] = 'primitive'
        descr = hyperparams.get('descr', '')
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        hyperparams['policy_opt']['share_buffer'] = True
        hyperparams['policy_opt']['buffers'] = hyperparams['buffers']
        hyperparams['policy_opt']['buffer_sizes'] = hyperparams['buffer_sizes']
        server = PolicyServer(hyperparams)
        server.agent = hyperparams['agent']['type'](hyperparams['agent'])
        ll_dir = hyperparams['ll_policy']
        hl_dir = hyperparams['hl_data']
        print(('Launching hl retrain from', ll_dir, hl_dir))
        #hl_retrain.retrain_hl_from_samples(server, hl_dir)
        server.data_gen.load_from_dir(LOG_DIR+hl_dir)
        server.run()


    def cont_only_retrain(self):
        hyperparams = self.config
        hyperparams['id'] = 'test'
        hyperparams['scope'] = 'cont'
        descr = hyperparams.get('descr', '')
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        hyperparams['policy_opt']['share_buffer'] = True
        hyperparams['policy_opt']['buffers'] = hyperparams['buffers']
        hyperparams['policy_opt']['buffer_sizes'] = hyperparams['buffer_sizes']
        server = PolicyServer(hyperparams)
        server.agent = hyperparams['agent']['type'](hyperparams['agent'])
        ll_dir = hyperparams['ll_policy']
        hl_dir = hyperparams['hl_data']
        print(('Launching hl retrain from', ll_dir, hl_dir))
        #hl_retrain.retrain_hl_from_samples(server, hl_dir)
        server.data_gen.load_from_dir(LOG_DIR+hl_dir)
        server.run()


    def hl_retrain(self, hyperparams):
        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['share_buffers'] = True
        hyperparams['id'] = 'test'
        descr = hyperparams.get('descr', '')
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        server = RolloutServer(hyperparams)
        ll_dir = hyperparams['ll_policy']
        hl_dir = hyperparams['hl_data']
        hl_retrain.retrain(server, hl_dir, ll_dir)


    def run_test(self, hyperparams):
        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['check_precond'] = False
        hyperparams['share_buffers'] = False
        hyperparams['load_render'] = True
        #hyperparams['agent']['image_height']  = 256
        #hyperparams['agent']['image_width']  = 256
        descr = hyperparams.get('descr', '')
        # hyperparams['weight_dir'] = hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
        hyperparams['id'] = 'test'
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        hyperparams['policy_opt']['share_buffer'] = True
        hyperparams['policy_opt']['buffers'] = hyperparams['buffers']
        hyperparams['policy_opt']['buffer_sizes'] = hyperparams['buffer_sizes']
        server = RolloutServer(hyperparams)
        newdir = 'experiment_logs/'+hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
        if not os.path.isdir(newdir):
            os.mkdir(newdir)
        server.hl_test_log = newdir + '/hl_test_rerun_log.npy'
        # if not os.path.isdir('tf_saved/'+hyperparams['weight_dir']+'_testruns'):
        #     os.mkdir('tf_saved/'+hyperparams['weight_dir']+'_testruns')
        # server.hl_test_log = 'tf_saved/' + hyperparams['weight_dir'] + '_testruns/hl_test_rerun_log.npy'
        ind = 0

        no = hyperparams['num_objs']
        n_vids = 20
        for test_run in range(hyperparams['num_tests']):
            print('RUN:', test_run)
            server.agent.replace_cond(0)
            server.agent.reset(0)
            server.test_hl(save=True, save_video=test_run<n_vids, save_fail=False)
        server.check_hl_statistics()
        '''
        while server.policy_opt.restore_ckpts(ind):
            for _ in range(50):
                server.agent.replace_cond(0)
                server.test_hl(5, save=True, ckpt_ind=ind)
            ind += 1
        '''
        sys.exit(0)

