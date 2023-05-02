""" This file defines policy optimization for a tensorflow policy. """
import copy
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.torch_models import *

MAX_UPDATE_SIZE = 10000
SCOPE_LIST = ['primitive', 'cont']
MODEL_DIR = 'saved_models/'


class TorchPolicyOpt():
    def __init__(self, hyperparams):
        self.config = hyperparams
        self.scope = self.config.get('scope', None)
        self.split_nets = self.config.get('split_nets', False)
        self.task_list = list(self.config['task_list'])
        self.ctrl_scopes = ['control'] if not self.split_nets else list(self.config['task_list'])
        self.torch_iter = 0
        self.batch_size = self.config['batch_size']
        self.load_all = self.config.get('load_all', False)
        self.share_buffers = self.config.get('share_buffer', True)
        if self.config.get('share_buffer', True):
            self.buffers = self.config['buffers']
            self.buf_sizes = self.config['buffer_sizes']

        self._primBounds = self.config['hl_network_params']['output_boundaries']
        self._contBounds = self.config['cont_network_params']['output_boundaries']
        self._dCtrl = self.config.get('dU')
        self._dPrim = max([b[1] for b in self._primBounds]) if len(self._primBounds) else 0
        self._dCont = max([b[1] for b in self._contBounds]) if len(self._contBounds) else 0
        self._dO = self.config.get('dO')
        self._dPrimObs = self.config.get('dPrimObs')
        self._dContObs = self.config.get('dContObs')

        self.device = torch.device('cpu')
        if self.config['use_gpu'] and torch.cuda.is_available():
            gpu_id = self.config['gpu_id']
            self.device = torch.device('cuda:{}'.format(gpu_id))
            self.gpu_fraction = self.config['gpu_fraction']
            torch.cuda.set_per_process_memory_fraction(self.gpu_fraction, device=self.device)

        self.init_networks()
        self.init_solvers()
        self._load_scopes()

        self.weight_dir = self.config['weight_dir']
        self.last_pkl_t = time.time()
        self.cur_pkl = 0
        self.update_count = 0
        if self.scope in ['primitive', 'cont']:
            self.update_size = self.config['prim_update_size']
        else:
            self.update_size = self.config['update_size']

        #self.update_size *= (1 + self.config.get('permute_hl', 0))

        self.train_iters = 0
        self.average_losses = []
        self.average_val_losses = []
        self.average_error = []
        self.N = 0
        self.n_updates = 0
        self.lr_scale = 0.9975
        self.lr_policy = 'fixed'
        self.config['iterations'] = MAX_UPDATE_SIZE // self.batch_size + 1
    
    def _load_scopes(self):
        llpol = self.config.get('ll_policy', '')
        hlpol = self.config.get('hl_policy', '')
        contpol = self.config.get('cont_policy', '')
        scopes = self.ctrl_scopes + SCOPE_LIST if self.scope is None else [self.scope]
        for scope in scopes:
            if len(llpol) and scope in self.ctrl_scopes:
                self.restore_ckpt(scope, dirname=llpol)
            if len(hlpol) and scope not in self.ctrl_scopes:
                self.restore_ckpt(scope, dirname=hlpol)
            if len(contpol) and scope not in self.ctrl_scopes:
                self.restore_ckpt(scope, dirname=contpol)


    def _set_opt(self, task):
        opt_cls = self.config.get('opt_cls', optim.Adam)
        if type(opt_cls) is str: opt_cls = getattr(optim, opt_cls)
        lr = self.config.get('lr', 1e-3)
        self.opts[task] = opt_cls(self.nets[task].parameters(), lr=lr) 


    def get_loss(self, task, x, y, precision=None,):
        model = self.nets[task]
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x, device=model.device)

        if type(y) is not torch.Tensor:
            y = torch.Tensor(y, device=model.device)

        if type(precision) is not torch.Tensor:
            precision = torch.Tensor(precision, device=model.device)

        pred = model.forward(x)

        return model.compute_loss(pred, y, precision)


    def train_step(self, task, x, y, precision=None):
        if task not in self.opts is None: self._set_opt(task)

        self.opts[task].zero_grad()
        loss = self.get_loss(task, x, y, precision)
        loss.backward()
        self.opts[task].step()
        return loss.item()


    def update(self, task="control", check_val=False, aux=[]):
        start_t = time.time()
        average_loss = 0
        for idx, batch in enumerate(self.data_loader):
            if idx >= self.config['iterations']:
                break

            x, y, precision = batch
            train_loss = self.train_step(task, x, y, precision)
            average_loss += train_loss
            self.torch_iter += 1
            self.N += len(batch)

        self.average_losses.append(average_loss / self.config['iterations'])


    def restore_ckpts(self, label=None):
        success = False
        for scope in self.ctrl_scopes + SCOPE_LIST:
            success = success or self.restore_ckpt(scope, label)
        return success


    def restore_ckpt(self, scope, label=None, dirname=''):
        ext = '' if label is None else '_{0}'.format(label)
        success = True
        if not len(dirname):
            dirname = self.weight_dir
        try:
            if dirname[-1] == '/':
                dirname = dirname[:-1]
           
            model = self.nets[scope]
            save_path = 'saved_models/'+dirname+'/'+scope+'{0}.ckpt'.format(ext)
            model.load_state_dict(torch.load(save_path))
            if scope in self.ctrl_scopes:
                self.nets[scope].scale = np.load(MODEL_DIR+dirname+'/'+scope+'_scale{0}.npy'.format(ext))
                self.nets[scope].bias = np.load(MODEL_DIR+dirname+'/'+scope+'_bias{0}.npy'.format(ext))
                
            self.write_shared_weights([scope])
            print(('Restored', scope, 'from', dirname))

        except Exception as e:
            print(('Could not restore', scope, 'from', dirname))
            print(e)
            success = False

        return success


    def write_shared_weights(self, scopes=None):
        if scopes is None: scopes = self.ctrl_scopes + SCOPE_LIST

        for scope in scopes:
            wts = self.serialize_weights([scope], save=True)
            with self.buf_sizes[scope].get_lock():
                self.buf_sizes[scope].value = len(wts)
                self.buffers[scope][:len(wts)] = wts


    def read_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.ctrl_scopes + SCOPE_LIST

        for scope in scopes:
            start_t = time.time()
            skip = False
            with self.buf_sizes[scope].get_lock():
                if self.buf_sizes[scope].value == 0: skip = True
                wts = self.buffers[scope][:self.buf_sizes[scope].value]

            wait_t = time.time() - start_t
            if wait_t > 0.1 and scope == 'primitive': print('Time waiting on model weights lock:', wait_t)
            if skip: continue

            try:
                self.deserialize_weights(wts)

            except Exception as e:
                if not skip:
                    traceback.print_exception(*sys.exc_info())
                    print(e)
                    print('Could not load {0} weights from {1}'.format(scope, self.scope))


    def serialize_weights(self, scopes=None, save=False):
        if scopes is None:
            all_scopes = self.ctrl_scopes + SCOPE_LIST
            ctrl_scopes = self.ctrl_scopes
        else:
            all_scopes = scopes
            ctrl_scopes = [scope for scope in scopes if scope in self.ctrl_scopes]

        models = {scope: self.nets[scope].state_dict() for scope in all_scopes if scope in self.nets}
        scales = {scope: self.nets[scope].scale.tolist() for scope in ctrl_scopes if scope in self.nets}
        biases = {scope: self.nets[scope].bias.tolist() for scope in ctrl_scopes if scope in self.nets}

        scales[''] = []
        biases[''] = []
        if save: self.store_scope_weights(scopes=scopes)
        return pickle.dumps([scopes, models, scales, biases])


    def deserialize_weights(self, json_wts, save=False):
        scopes, models, scales, biases = pickle.loads(json_wts)

        for scope in scopes:
            self.nets[scope].load_state_dict(models[scope])

            if scope in self.ctrl_scopes:
                self.nets[scope].scale = np.array(scales[scope])
                self.nets[scope].bias = np.array(biases[scope])

        if save: self.store_scope_weights(scopes=scopes)


    def update_weights(self, scope, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        model = self.nets[scope]
        save_path = MODEL_DIR + weight_dir+'/'+scope+'.ckpt'
        model.load_state_dict(torch.load(save_path))


    def store_scope_weights(self, scopes, weight_dir=None, lab=''):
        if weight_dir is None:
            weight_dir = self.weight_dir

        for scope in scopes:
            model = self.nets[scope]
            try:
                save_path = MODEL_DIR + weight_dir+'/'+scope+'.ckpt'
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(model.state_dict(), save_path)

            except:
                print('Saving torch model encountered an issue but it will not crash:')
                traceback.print_exception(*sys.exc_info())

        if scope in self.ctrl_scopes:
            policy = self.nets[scope]
            scale_bias_save_dir = MODEL_DIR+weight_dir
            if not os.path.isdir(scale_bias_save_dir):
                os.mkdirs(scale_bias_save_dir)
            np.save(scale_bias_save_dir+'/'+'_scale{0}'.format(lab), policy.scale)
            np.save(scale_bias_save_dir+'/'+scope+'_bias{0}'.format(lab), policy.bias)


    def store_weights(self, weight_dir=None):
        if self.scope is None:
            self.store_scope_weights(self.ctrl_scopes+SCOPE_LIST, weight_dir)
        else:
            self.store_scope_weights([self.scope], weight_dir)


    def update_lr(self):
        if self.method == 'linear':
            self.cur_lr *= self.lr_scale
            self.cur_hllr *= self.lr_scale

    def select_dims(self, scope):
        dO = self._dO
        if scope == 'primitive':
            dO = self._dPrimObs
        if scope == 'cont':
            dO = self._dContObs

        dU = self._dCtrl
        if scope == 'primitive':
            dU = self._dPrim
        if scope == 'cont':
            dU = self._dCont

        return dO, dU


    def _init_network(self, scope):
        config = self.config['ll_network_params']
        if 'primitive' == scope:
            config = self.config['hl_network_params']
        elif 'cont' == scope:
            config = self.config['cont_network_params']

        self.nets[scope] = PolicyNet(config=config,
                                     scope=scope,
                                     device=self.device)
        self.nets[scope].to_device(self.device)

    def init_networks(self):
        """ Helper method to initialize the tf networks used """
        self.nets = {}
        scopes = self.ctrl_scopes + SCOPE_LIST if (self.scope is None or self.load_all) else [self.scope]
        for scope in scopes:
            self._init_network(scope)
                
        else:
            self._init_network(self.scope)


    def init_solvers(self):
        """ Helper method to initialize the solver. """
        self.opts = {}
        self.cur_dec = self.config['weight_decay']
        scopes = self.ctrl_scopes + SCOPE_LIST if self.scope is None else [self.scope]
        for scope in scopes:
            self._set_opt(scope)


    def get_policy(self, task):
        if task in self.nets: 
            return self.nets[task]

        elif task in self.task_list:
            return self.nets['control']

        else:
            raise ValueError('Cannot find policy for {}'.format(task))
 

    def policy_initialized(self, task):
        policy = self.get_policy(task)
        return policy.is_initialized()


    def task_distr(self, prim_obs, eta=1.):
        return self.nets["primitive"].task_distr(prim_obs, bounds=self._primBounds)

    def cont_task(self, prim_obs, eta=1.):
        return self.nets["cont"].task_distr(prim_obs, bounds=self._contBounds)

    def task_acc(self, obs, tgt_mu, prc, piecewise=False, scalar=True):
        acc = []
        task = 'primitive'
        for n in range(len(obs)):
            distrs = self.nets[task].task_distr(obs[n], bounds=self._primBounds)
            labels = []
            for bound in self._primBounds:
                labels.append(tgt_mu[n, bound[0]:bound[1]])
            accs = []
            for i in range(len(labels)):
                #if prc[n][i] < 1e-3 or np.abs(np.max(labels[i])-np.min(labels[i])) < 1e-2:
                #    accs.append(1)
                #    continue

                if np.argmax(distrs[i]) != np.argmax(labels[i]):
                    accs.append(0)
                else:
                    accs.append(1)

            if piecewise or not scalar:
                acc.append(accs)
            else:
                acc.append(np.min(accs) * np.ones(len(accs)))
            #acc += np.mean(accs) if piecewise else np.min(accs)
        if scalar:
            return np.mean(acc)
        return np.mean(acc, axis=0)


    def check_task_error(self, obs, mu):
        err = 0.
        for o in obs:
            distrs = self.nets['primitive'].task_distr(o, bounds=self._primBounds)
            i = 0
            for d in distrs:
                ind1 = np.argmax(d)
                ind2 = np.argmax(mu[i:i+len(d)])
                if ind1 != ind2: err += 1./len(distrs)
                i += len(d)
        err /= len(obs)
        self.average_error.append(err)
        return err


    def check_validation(self, obs, tgt_mu, tgt_prc, task="control"):
        return [self.get_loss(task, obs, tgt_mu, tgt_prc).item()]


