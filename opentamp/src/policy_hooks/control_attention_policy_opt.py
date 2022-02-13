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
SCOPE_LIST = ['primitive', 'cont', 'label']


class PolicyOpt(object):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU, dPrimObs, dContObs, dValObs, primBounds, contBounds=None, inputs=None):
        self.scope = hyperparams['scope'] if 'scope' in hyperparams else None
        self.config = hyperparams
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

        self._dPrim = max([b[1] for b in primBounds])
        self._dCont = max([b[1] for b in contBounds]) if contBounds is not None and len(contBounds) else 0
        self._dPrimObs = dPrimObs
        self._dContObs = dContObs
        self._primBounds = primBounds
        self._contBounds = contBounds if contBounds is not None else []
        self._compute_idx()

        self.device = torch.device('cpu')
        if self._hyperparams['use_gpu'] == 1:
            gpu_id = self._hyperparams['gpu_id']
            self.device = torch.device('cuda:{}'.format(gpu_id)
        self.gpu_fraction = self._hyperparams['gpu_fraction']
        torch.cuda.set_per_process_memory_fraction(self.gpu_fraction, device=self.device)

        self.init_networks()
        self.init_solvers()

        self.weight_dir = self._hyperparams['weight_dir']
        self.last_pkl_t = time.time()
        self.cur_pkl = 0

        self.init_policies(dU)
        llpol = hyperparams.get('ll_policy', '')
        hlpol = hyperparams.get('hl_policy', '')
        contpol = hyperparams.get('cont_policy', '')
        scopes = self.valid_scopes + SCOPE_LIST if self.scope is None else [self.scope]
        for scope in scopes:
            if len(llpol) and scope in self.valid_scopes:
                self.restore_ckpt(scope, dirname=llpol)
            if len(hlpol) and scope not in self.valid_scopes:
                self.restore_ckpt(scope, dirname=hlpol)
            if len(contpol) and scope not in self.valid_scopes:
                self.restore_ckpt(scope, dirname=contpol)

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


 
    def _set_opt(self, config, task):
        opt_cls = config.get('opt_cls', optim.Adam)
        if type(opt_cls) is str: opt_cls = getattr(optim, opt_cls)
        lr = config.get('lr', 1e-3)
        self.opts[task] = opt_cls(self.nets[task].parameters(), lr=lr) 


    def train_step(self, x, y, precision=None):
        if self.opt is None: self._set_opt()
        (x, y) = (x.to(self.device), y.to(self.device))
        pred = self(x)
        if precision is None:
            loss = self.loss_fn(pred, y, reduction='mean')
        elif precision.size()[-1] > 1:
            loss = self.loss_fn(pred, y, precision=precision)
            loss = torch.mean(loss)
        else:
            loss = self.loss_fn(pred, y, reduction='none') * precision
            loss = torch.mean(loss)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


    def update(self, task="control", check_val=False, aux=[]):
        start_t = time.time()
        average_loss = 0
        for i in range(self._hyperparams['iterations']):
            x, y, precision = self.data_loader.next()
            train_loss = self.train_step()
            average_loss += train_loss
            self.tf_iter += 1
        self.average_losses.append(average_loss / self._hyperparams['iterations'])


    def restore_ckpts(self, label=None):
        success = False
        for scope in self.valid_scopes + SCOPE_LIST:
            success = success or self.restore_ckpt(scope, label)
        return success


    def restore_ckpt(self, scope, label=None, dirname=''):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/')
        if not len(variables): return False
        self.saver = tf.train.Saver(variables)
        ext = ''
        if label is not None:
            ext = '_{0}'.format(label)
        success = True
        if not len(dirname):
            dirname = self.weight_dir
        try:
            if dirname[-1] == '/':
                dirname = dirname[:-1]
            self.saver.restore(self.sess, 'tf_saved/'+dirname+'/'+scope+'{0}.ckpt'.format(ext))
            if scope in self.task_map:
                self.task_map[scope]['policy'].scale = np.load('tf_saved/'+dirname+'/'+scope+'_scale{0}.npy'.format(ext))
                self.task_map[scope]['policy'].bias = np.load('tf_saved/'+dirname+'/'+scope+'_bias{0}.npy'.format(ext))
                #self.var[scope] = np.load('tf_saved/'+dirname+'/'+scope+'_variance{0}.npy'.format(ext))
                #self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(self.var[scope]))
            self.write_shared_weights([scope])
            print(('Restored', scope, 'from', dirname))
        except Exception as e:
            print(('Could not restore', scope, 'from', dirname))
            print(e)
            success = False

        return success


    def write_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

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
            if wait_t > 0.1 and scope == 'primitive': print('Time waiting on lock:', wait_t)
            #if self.buf_sizes[scope].value == 0: skip = True
            #wts = self.buffers[scope][:self.buf_sizes[scope].value]

            if skip: continue
            try:
                self.deserialize_weights(wts)
            except Exception as e:
                #traceback.print_exception(*sys.exc_info())
                if not skip:
                    print(e)
                    print('Could not load {0} weights from {1}'.format(scope, self.scope), e)


    def serialize_weights(self, scopes=None, save=True):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

        var_to_val = {}
        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/')
            for v in variables:
                var_to_val[v.name] = self.sess.run(v).tolist()

        scales = {task: self.task_map[task]['policy'].scale.tolist() for task in scopes if task in self.task_map}
        biases = {task: self.task_map[task]['policy'].bias.tolist() for task in scopes if task in self.task_map}
        if hasattr(self, 'prim_policy') and 'primitive' in scopes:
            scales['primitive'] = self.prim_policy.scale.tolist()
            biases['primitive'] = self.prim_policy.bias.tolist()

        if hasattr(self, 'cont_policy') and 'cont' in scopes:
            scales['cont'] = self.cont_policy.scale.tolist()
            biases['cont'] = self.cont_policy.bias.tolist()

        #variances = {task: self.var[task].tolist() for task in scopes if task in self.task_map}
        variances = {}
        scales[''] = []
        biases[''] = []
        variances[''] = []
        if save: self.store_scope_weights(scopes=scopes)
        return pickle.dumps([scopes, var_to_val, scales, biases, variances])


    def deserialize_weights(self, json_wts, save=False):
        scopes, var_to_val, scales, biases, variances = pickle.loads(json_wts)

        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/')
            for var in variables:
                var.load(var_to_val[var.name], session=self.sess)

            if scope == 'primitive' and hasattr(self, 'prim_policy'):
                self.prim_policy.scale = np.array(scales[scope])
                self.prim_policy.bias = np.array(biases[scope])

            if scope == 'cont' and hasattr(self, 'cont_policy'):
                self.cont_policy.scale = np.array(scales[scope])
                self.cont_policy.bias = np.array(biases[scope])

            if scope not in self.valid_scopes: continue
            # if save:
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_scale', scales['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_bias', biases['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_variance', variances['control'])
            #self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(np.array(variances[scope])))
            self.task_map[scope]['policy'].scale = np.array(scales[scope])
            self.task_map[scope]['policy'].bias = np.array(biases[scope])
            #self.var[scope] = np.array(variances[scope])
        if save: self.store_scope_weights(scopes=scopes)

    def update_weights(self, scope, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        self.saver.restore(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')

    def store_scope_weights(self, scopes, weight_dir=None, lab=''):
        if weight_dir is None:
            weight_dir = self.weight_dir
        for scope in scopes:
            try:
                variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/')
                saver = tf.train.Saver(variables)
                saver.save(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'{0}.ckpt'.format(lab))
            except:
                print('Saving variables encountered an issue but it will not crash:')
                traceback.print_exception(*sys.exc_info())

        if scope in self.task_map:
            policy = self.task_map[scope]['policy']
            np.save('tf_saved/'+weight_dir+'/'+scope+'_scale{0}'.format(lab), policy.scale)
            np.save('tf_saved/'+weight_dir+'/'+scope+'_bias{0}'.format(lab), policy.bias)
            #np.save('tf_saved/'+weight_dir+'/'+scope+'_variance{0}'.format(lab), self.var[scope])

    def store_weights(self, weight_dir=None):
        if self.scope is None:
            self.store_scope_weights(self.valid_scopes+SCOPE_LIST, weight_dir)
        else:
            self.store_scope_weights([self.scope], weight_dir)

    def get_data(self):
        return [self.mu, self.obs, self.prc, self.wt, self.val_mu, self.val_obs, self.val_prc, self.val_wt]


    def update_lr(self):
        if self.method == 'linear':
            self.cur_lr *= self.lr_scale
            self.cur_hllr *= self.lr_scale


    def _create_network(self, name, info):
        with tf.variable_scope(name):
            self.etas[name] = tf.placeholder_with_default(1., shape=())
            tf_map_generator = info['network_model']
            info['network_params']['eta'] = self.etas[name]
            #self.class_tensors[name] = tf.placeholder(shape=[None, 1], dtype='float32')
            tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=info['dO'], \
                                                               dim_output=info['dOut'], \
                                                               batch_size=info['batch_size'], \
                                                               network_config=info['network_params'], \
                                                               input_layer=info['input_layer'])

            self.obs_tensors[name] = tf_map.get_input_tensor()
            self.precision_tensors[name] = tf_map.get_precision_tensor()
            self.action_tensors[name] = tf_map.get_target_output_tensor()
            self.act_ops[name] = tf_map.get_output_op()
            self.feat_ops[name] = tf_map.get_feature_op()
            self.loss_scalars[name] = tf_map.get_loss_op()
            self.fc_vars[name] = fc_vars
            self.last_conv_vars[name] = last_conv_vars


    def init_network(self):
        """ Helper method to initialize the tf networks used """
        self.nets = {}
        if self.load_all or self.scope is None:
            for scope in self.valid_scopes:
                self.nets[scope] = PolicyNet(self._hyperparams['network_model'], device=self.device)
                
        else:
            config = self._hyperparams['network_model']
            if 'primitive' == self.scope: config = self._hyperparams['primitive_network_model']

        input_tensor = None
        if self.load_all or self.scope is None or 'primitive' == self.scope:
            with tf.variable_scope('primitive'):
                inputs = self.input_layer if 'primitive' == self.scope else None
                self.primitive_eta = tf.placeholder_with_default(1., shape=())
                tf_map_generator = self._hyperparams['primitive_network_model']
                self.primitive_class_tensor = None
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, \
                                                                   dim_output=self._dPrim, \
                                                                   batch_size=self.batch_size, \
                                                                   network_config=self._hyperparams['primitive_network_params'], \
                                                                   input_layer=inputs, \
                                                                   eta=self.primitive_eta)
                self.primitive_obs_tensor = tf_map.get_input_tensor()
                self.primitive_precision_tensor = tf_map.get_precision_tensor()
                self.primitive_action_tensor = tf_map.get_target_output_tensor()
                self.primitive_act_op = tf_map.get_output_op()
                self.primitive_feat_op = tf_map.get_feature_op()
                self.primitive_loss_scalar = tf_map.get_loss_op()
                self.primitive_fc_vars = fc_vars
                self.primitive_last_conv_vars = last_conv_vars
                self.primitive_aux_losses = tf_map.aux_loss_ops

                # Setup the gradients
                #self.primitive_grads = [tf.gradients(self.primitive_act_op[:,u], self.primitive_obs_tensor)[0] for u in range(self._dPrim)]

        if (self.load_all or self.scope is None or 'cont' == self.scope) and len(self._contBounds):
            with tf.variable_scope('cont'):
                inputs = self.input_layer if 'cont' == self.scope else None
                self.cont_eta = tf.placeholder_with_default(1., shape=())
                tf_map_generator = self._hyperparams['cont_network_model']
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dContObs, \
                                                                   dim_output=self._dCont, \
                                                                   batch_size=self.batch_size, \
                                                                   network_config=self._hyperparams['cont_network_params'], \
                                                                   input_layer=inputs, \
                                                                   eta=self.cont_eta)
                self.cont_obs_tensor = tf_map.get_input_tensor()
                self.cont_precision_tensor = tf_map.get_precision_tensor()
                self.cont_action_tensor = tf_map.get_target_output_tensor()
                self.cont_act_op = tf_map.get_output_op()
                self.cont_feat_op = tf_map.get_feature_op()
                self.cont_loss_scalar = tf_map.get_loss_op()
                self.cont_fc_vars = fc_vars
                self.cont_last_conv_vars = last_conv_vars
                self.cont_aux_losses = tf_map.aux_loss_ops

        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                with tf.variable_scope(scope):
                    self.task_map[scope] = {}
                    tf_map_generator = self._hyperparams['network_model']
                    tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, \
                                                                       dim_output=self._dU, \
                                                                       batch_size=self.batch_size, \
                                                                       network_config=self._hyperparams['network_params'], \
                                                                       input_layer=self.input_layer)
                    self.task_map[scope]['obs_tensor'] = tf_map.get_input_tensor()
                    self.task_map[scope]['precision_tensor'] = tf_map.get_precision_tensor()
                    self.task_map[scope]['action_tensor'] = tf_map.get_target_output_tensor()
                    self.task_map[scope]['act_op'] = tf_map.get_output_op()
                    self.task_map[scope]['feat_op'] = tf_map.get_feature_op()
                    self.task_map[scope]['loss_scalar'] = tf_map.get_loss_op()
                    self.task_map[scope]['fc_vars'] = fc_vars
                    self.task_map[scope]['last_conv_vars'] = last_conv_vars

                    # Setup the gradients
                    #self.task_map[scope]['grads'] = [tf.gradients(self.task_map[scope]['act_op'][:,u], self.task_map[scope]['obs_tensor'])[0] for u in range(self._dU)]
        
        if (self.scope is None or 'label' == self.scope) and self.load_label:
            with tf.variable_scope('label'):
                inputs = self.input_layer if 'label' == self.scope else None
                self.label_eta = tf.placeholder_with_default(1., shape=())
                tf_map_generator = self._hyperparams['primitive_network_model']
                self.label_class_tensor = None
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, \
                                                                   dim_output=2, \
                                                                   batch_size=self.batch_size, \
                                                                   network_config=self._hyperparams['label_network_params'], \
                                                                   input_layer=inputs, \
                                                                   eta=self.label_eta)
                self.label_obs_tensor = tf_map.get_input_tensor()
                self.label_precision_tensor = tf_map.get_precision_tensor()
                self.label_action_tensor = tf_map.get_target_output_tensor()
                self.label_act_op = tf_map.get_output_op()
                self.label_feat_op = tf_map.get_feature_op()
                self.label_loss_scalar = tf_map.get_loss_op()
                self.label_fc_vars = fc_vars
                self.label_last_conv_vars = last_conv_vars
                self.label_aux_losses = tf_map.aux_loss_ops

                # Setup the gradients
                #self.primitive_grads = [tf.gradients(self.primitive_act_op[:,u], self.primitive_obs_tensor)[0] for u in range(self._dPrim)]


    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.opts = {}
        self.cur_dec = self._hyperparams['weight_decay']
        if self.scope is not None:
            self._set_opt(self.config, self.scope)
        else:
            for scope in self.scopes:
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

        if self.load_label and (self.scope is None or self.scope == 'label'):
            self.label_policy = TfPolicy(2,
                                        self.label_obs_tensor,
                                        self.label_act_op,
                                        self.label_feat_op,
                                        np.zeros(2),
                                        self.sess,
                                        self.device_string,
                                        copy_param_scope=None,
                                        normalize=False)
 

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
        if task == 'primitive':
            feed_dict = {self.primitive_obs_tensor: obs,
                         self.primitive_action_tensor: tgt_mu,
                         self.primitive_precision_tensor: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, train=False)
        elif task == 'cont':
            feed_dict = {self.cont_obs_tensor: obs,
                         self.cont_action_tensor: tgt_mu,
                         self.cont_precision_tensor: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.cont_solver(feed_dict, self.sess, device_string=self.device_string, train=False)
        elif task == 'label':
            feed_dict = {self.label_obs_tensor: obs,
                         self.label_action_tensor: tgt_mu,
                         self.label_precision_tensor: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.label_solver(feed_dict, self.sess, device_string=self.device_string, train=False)
        else:
            feed_dict = {self.task_map[task]['obs_tensor']: obs,
                         self.task_map[task]['action_tensor']: tgt_mu,
                         self.task_map[task]['precision_tensor']: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string, train=False)
        #self.average_val_losses.append(val_loss)
        return val_loss


    def policy_initialized(self, task):
        if task in self.valid_scopes:
            return self.task_map[task]['policy'].scale is not None
        return self.task_map['control']['policy'].scale is not None

    def save_model(self, fname):
        # LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        # LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': {task:self.task_map[task]['policy'].scale for task in self.task_map},
            'bias': {task:self.task_map[task]['policy'].bias for task in self.task_map},
            'tf_iter': self.tf_iter,
            'x_idx': {task:self.task_map[task]['policy'].x_idx for task in self.task_map},
            'chol_pol_covar': {task:self.task_map[task]['policy'].chol_pol_covar for task in self.task_map},
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tf.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        for task in self.task_map:
            self.policy[task].scale = state['scale']
            self.policy[task].bias = state['bias']
            self.policy[task].x_idx = state['x_idx']
            self.policy[task].chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)
