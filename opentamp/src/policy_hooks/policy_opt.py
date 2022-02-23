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
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.tf_policy import TfPolicy
from policy_hooks.torch_models import *

MAX_UPDATE_SIZE = 10000
SCOPE_LIST = ['primitive', 'cont']
MODEL_DIR = 'saved_models/'


class PolicyOpt(object):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, inputs=None):
        self.config = hyperparams
        self.scope = hyperparams.get('scope', None)
        self.split_nets = hyperparams.get('split_nets', False)
        self.valid_scopes = ['control'] if not self.split_nets else list(config['task_list'])
        self.torch_iter = 0
        self.batch_size = self._hyperparams['batch_size']
        self.load_all = self._hyperparams.get('load_all', False)
        self.input_layer = inputs
        self.share_buffers = self._hyperparams.get('share_buffer', True)
        if self._hyperparams.get('share_buffer', True):
            self.buffers = self._hyperparams['buffers']
            self.buf_sizes = self._hyperparams['buffer_sizes']

        self._primBounds = hyperparams.get('prim_bounds', [(0,0)])
        self._contBounds = hyperparams.get('cont_bounds', [(0,0)])
        self._dCtrl = hyperparams.get.get('dU')
        self._dPrim = max([b[1] for b in self._primBounds])
        self._dCont = max([b[1] for b in self._contBounds])
        self._dO = hyperparams.get('dO', None)
        self._dPrimObs = hyperparams.get('dPrimObs', None)
        self._dContObs = hyperparams.get('dContObs', None)
        self._compute_idx()

        self.device = torch.device('cpu')
        if self._hyperparams['use_gpu'] == 1:
            gpu_id = self._hyperparams['gpu_id']
            self.device = torch.device('cuda:{}'.format(gpu_id))
        self.gpu_fraction = self._hyperparams['gpu_fraction']
        torch.cuda.set_per_process_memory_fraction(self.gpu_fraction, device=self.device)
        self.init_networks()
        self.init_solvers()
        self.init_policies()
        self._load_scopes()

        self.weight_dir = self._hyperparams['weight_dir']
        self.last_pkl_t = time.time()
        self.cur_pkl = 0
        self.update_count = 0
        if self.scope in ['primitive', 'cont']:
            self.update_size = self._hyperparams['prim_update_size']
        else:
            self.update_size = self._hyperparams['update_size']

        #self.update_size *= (1 + self._hyperparams.get('permute_hl', 0))

        self.train_iters = 0
        self.average_losses = []
        self.average_val_losses = []
        self.average_error = []
        self.N = 0
        self.n_updates = 0
        self.lr_scale = 0.9975
        self.lr_policy = 'fixed'
        self._hyperparams['iterations'] = MAX_UPDATE_SIZE // self.batch_size + 1

    
    def _load_scopes(self):
        llpol = self.config.get('ll_policy', '')
        hlpol = self.config.get('hl_policy', '')
        contpol = self.config.get('cont_policy', '')
        scopes = self.valid_scopes + SCOPE_LIST if self.scope is None else [self.scope]
        for scope in scopes:
            if len(llpol) and scope in self.valid_scopes:
                self.restore_ckpt(scope, dirname=llpol)
            if len(hlpol) and scope not in self.valid_scopes:
                self.restore_ckpt(scope, dirname=hlpol)
            if len(contpol) and scope not in self.valid_scopes:
                self.restore_ckpt(scope, dirname=contpol)


    def _compute_idx(self):
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params'].get('obs_image_data', []):
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim

        self.prim_x_idx, self.prim_img_idx, i = [], [], 0
        for sensor in self._hyperparams['primitive_network_params']['obs_include']:
            dim = self._hyperparams['primitive_network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['primitive_network_params'].get('obs_image_data', []):
                self.prim_img_idx = self.prim_img_idx + list(range(i, i+dim))
            else:
                self.prim_x_idx = self.prim_x_idx + list(range(i, i+dim))
            i += dim

        self.cont_x_idx, self.cont_img_idx, i = [], [], 0
        for sensor in self._hyperparams['cont_network_params']['obs_include']:
            dim = self._hyperparams['cont_network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['cont_network_params'].get('obs_image_data', []):
                self.cont_img_idx = self.cont_img_idx + list(range(i, i+dim))
            else:
                self.cont_x_idx = self.cont_x_idx + list(range(i, i+dim))
            i += dim


    def _set_opt(self, task):
        opt_cls = self._hyperparams.get('opt_cls', optim.Adam)
        if type(opt_cls) is str: opt_cls = getattr(optim, opt_cls)
        lr = self._hyperparams.get('lr', 1e-3)
        self.opts[task] = opt_cls(self.nets[task].parameters(), lr=lr) 


    def get_loss(self, task, x, y, precision=None):
        model = self.nets[task]
        target = model(x)
        if model.use_precision:
            if precision.size()[-1] > 1:
                return torch.mean(model.loss_fn(target, y, precision))
            else:
                sum_loss = torch.sum(model.loss_fn(pred, y, reduction='none') * precision)
                return sum_loss / torch.sum(precision)
        else:
            return model.loss_fn(target, y, reduction='mean')


    def train_step(self, task, x, y, precision=None):
        if task not in self.opts is None: self._set_opt(task)
        (x, y) = (x.to(self.device), y.to(self.device))
        model = self.nets[task]
        pred = model(x)
        loss = self.get_loss(task, x, y, precision)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


    def update(self, task="control", check_val=False, aux=[]):
        start_t = time.time()
        average_loss = 0
        for i in range(self._hyperparams['iterations']):
            x, y, precision = self.data_loader.next()
            train_loss = self.train_step(x, y, precision)
            average_loss += train_loss
            self.tf_iter += 1
        self.average_losses.append(average_loss / self._hyperparams['iterations'])


    def restore_ckpts(self, label=None):
        success = False
        for scope in self.valid_scopes + SCOPE_LIST:
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
            if scope in self.task_map:
                self.task_map[scope]['policy'].scale = np.load(MODEL_DIR+dirname+'/'+scope+'_scale{0}.npy'.format(ext))
                self.task_map[scope]['policy'].bias = np.load(MODEL_DIR+dirname+'/'+scope+'_bias{0}.npy'.format(ext))
            self.write_shared_weights([scope])
            print(('Restored', scope, 'from', dirname))

        except Exception as e:
            print(('Could not restore', scope, 'from', dirname))
            print(e)
            success = False

        return success


    def write_shared_weights(self, scopes=None):
        if scopes is None: scopes = self.valid_scopes + SCOPE_LIST

        for scope in scopes:
            wts = self.serialize_weights([scope])
            with self.buf_sizes[scope].get_lock():
                self.buf_sizes[scope].value = len(wts)
                self.buffers[scope][:len(wts)] = wts


    def read_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

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
                #traceback.print_exception(*sys.exc_info())
                if not skip:
                    print(e)
                    print('Could not load {0} weights from {1}'.format(scope, self.scope), e)


    def serialize_weights(self, scopes=None, save=False):
        if scopes is None: scopes = self.valid_scopes + SCOPE_LIST
        models = {scope: self.nets[scope].state_dict() for scope in scopes if scope in self.nets}
        scales = {task: self.task_map[task]['policy'].scale.tolist() for task in scopes if task in self.task_map}
        biases = {task: self.task_map[task]['policy'].bias.tolist() for task in scopes if task in self.task_map}

        if hasattr(self, 'prim_policy') and 'primitive' in scopes:
            scales['primitive'] = self.prim_policy.scale.tolist()
            biases['primitive'] = self.prim_policy.bias.tolist()

        if hasattr(self, 'cont_policy') and 'cont' in scopes:
            scales['cont'] = self.cont_policy.scale.tolist()
            biases['cont'] = self.cont_policy.bias.tolist()

        scales[''] = []
        biases[''] = []
        if save: self.store_scope_weights(scopes=scopes)
        return pickle.dumps([scopes, models, scales, biases])


    def deserialize_weights(self, json_wts, save=False):
        scopes, models, scales, biases = pickle.loads(json_wts)

        for scope in scopes:
            self.nets[scope].load_state_dict(models[scope])
            if scope == 'primitive' and hasattr(self, 'prim_policy'):
                self.prim_policy.scale = np.array(scales[scope])
                self.prim_policy.bias = np.array(biases[scope])

            if scope == 'cont' and hasattr(self, 'cont_policy'):
                self.cont_policy.scale = np.array(scales[scope])
                self.cont_policy.bias = np.array(biases[scope])

            if scope not in self.task_map: continue
            self.task_map[scope]['policy'].scale = np.array(scales[scope])
            self.task_map[scope]['policy'].bias = np.array(biases[scope])
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
                torch.save(the_model.state_dict(), save_path)

            except:
                print('Saving torch model encountered an issue but it will not crash:')
                traceback.print_exception(*sys.exc_info())

        if scope in self.task_map:
            policy = self.task_map[scope]['policy']
            np.save(MODEL_DIR+weight_dir+'/'+scope+'_scale{0}'.format(lab), policy.scale)
            np.save(MODEL_DIR+weight_dir+'/'+scope+'_bias{0}'.format(lab), policy.bias)


    def store_weights(self, weight_dir=None):
        if self.scope is None:
            self.store_scope_weights(self.valid_scopes+SCOPE_LIST, weight_dir)
        else:
            self.store_scope_weights([self.scope], weight_dir)


    def update_lr(self):
        if self.method == 'linear':
            self.cur_lr *= self.lr_scale
            self.cur_hllr *= self.lr_scale

    def _select_dims(self, scope):
        dO = self.dO
        if scope == 'primitive':
            dO = self.dPrimObs
        if scope == 'cont':
            dO = self.dContObs

        dU = self._dCtrl
        if scope == 'primitive':
            dU = self._dPrim
        if scope == 'cont':
            dU = self._dCont

        return dO, dU


    def init_networks(self):
        """ Helper method to initialize the tf networks used """
        self.nets = {}
        if self.load_all or self.scope is None:
            for scope in self.valid_scopes:
                dO, dU = self._select_dims(scope)
                config = self._hyperparams['network_model']
                if 'primitive' == self.scope: config = self._hyperparams['primitive_network_model']
                
                config['dim_input'] = dO
                config['dim_output'] = dU
                self.nets[scope] = PolicyNet(config=config,
                                             device=self.device)
                
        else:
            config = self._hyperparams['network_model']
            if 'primitive' == self.scope: config = self._hyperparams['primitive_network_model']
            dO, dU = self._select_dims(self.scope)
            config['dim_input'] = dO
            config['dim_output'] = dU
            self.nets[scope] = PolicyNet(config=config,
                                         device=self.device)


    def init_solvers(self):
        """ Helper method to initialize the solver. """
        self.opts = {}
        self.cur_dec = self._hyperparams['weight_decay']
        scopes = self.scopes if self.scope is None else [self.scope]
        for scope in scopes:
            self._set_opt(self.config, scope)


    def get_policy(self, task):
        if task == 'primitive': return self.prim_policy
        if task == 'cont': return self.cont_policy
        if task == 'label': return self.label_policy
        return self.task_map[task]['policy']


    def init_policies(self, dU):
        if self.load_all or self.scope is None or self.scope == 'primitive':
            self.prim_policy = TfPolicy(self._dPrim,
                                        self.primitive_obs_tensor,
                                        self.primitive_act_op,
                                        self.primitive_feat_op,
                                        np.zeros(self._dPrim),
                                        self.sess,
                                        self.device_string,
                                        copy_param_scope=None,
                                        normalize=False)
        if (self.load_all or self.scope is None or self.scope == 'cont') and len(self._contBounds):
            self.cont_policy = TfPolicy(self._dCont,
                                        self.cont_obs_tensor,
                                        self.cont_act_op,
                                        self.cont_feat_op,
                                        np.zeros(self._dCont),
                                        self.sess,
                                        self.device_string,
                                        copy_param_scope=None,
                                        normalize=False)
        for scope in self.valid_scopes:
            normalize = IM_ENUM not in self._hyperparams['network_params']['obs_include']
            if self.scope is None or scope == self.scope:
                self.task_map[scope]['policy'] = TfPolicy(dU,
                                                        self.task_map[scope]['obs_tensor'],
                                                        self.task_map[scope]['act_op'],
                                                        self.task_map[scope]['feat_op'],
                                                        np.zeros(dU),
                                                        self.sess,
                                                        self.device_string,
                                                        normalize=normalize,
                                                        copy_param_scope=None)
 

    def task_acc(self, obs, tgt_mu, prc, piecewise=False, scalar=True):
        acc = []
        task = 'primitive'
        for n in range(len(obs)):
            distrs = self.task_distr(obs[n])
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


    def cont_task(self, obs, eta=1.):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)

        vals = self.sess.run(self.cont_act_op, feed_dict={self.cont_obs_tensor:obs, self.cont_eta: eta, self.dec_tensor: self.cur_dec})[0].flatten()
        res = []
        for bound in self._contBounds:
            res.append(vals[bound[0]:bound[1]])
        return res


    def task_distr(self, obs, eta=1.):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)

        distr = self.sess.run(self.primitive_act_op, feed_dict={self.primitive_obs_tensor:obs, self.primitive_eta: eta, self.dec_tensor: self.cur_dec})[0].flatten()
        res = []
        for bound in self._primBounds:
            res.append(distr[bound[0]:bound[1]])
        return res


    def label_distr(self, obs, eta=1.):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)

        distr = self.sess.run(self.label_act_op, feed_dict={self.label_obs_tensor:obs, self.label_eta: eta, self.dec_tensor: self.cur_dec})[0]
        return distr


    def check_task_error(self, obs, mu):
        err = 0.
        for o in obs:
            distrs = self.task_distr(o)
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
        return self.get_loss(task, obs, tgt_mu, tgt_prc).item()


    def policy_initialized(self, task):
        if task in self.valid_scopes:
            return self.task_map[task]['policy'].scale is not None
        return self.task_map['control']['policy'].scale is not None


