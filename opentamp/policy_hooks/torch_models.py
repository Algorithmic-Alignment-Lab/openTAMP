""" This file provides an example tensorflow network used to define a policy. """

import functools
import operator
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def precision_mse(output, y, precision):
    return torch.matmul(torch.matmul(output, precision), y.transpose())


class TorchNet(nn.Module):
    def __init__(self, config, device=None):
        nn.Module.__init__(self)
        self.config = config

        if device is None: device = torch.device('cpu')
        if type(device) is str: device = torch.device(device)
        self.device = device

        self.conv_layers = []
        self.fc_layers = []

        self.fc_input_dim = 0
        for sensor in self.config['obs_include']:
            dim = self.config['sensor_dims'][sensor]
            self.fc_input_dim += dim

        self.output_dim = 0
        for sensor in self.config['out_include']:
            dim = self.config['sensor_dims'][sensor]
            self.output_dim += dim

        self._compute_idx()
        self._build_conv_layers()
        
        self.conv_to_fc = config.get('conv_to_fc', 'fp')
        if self.conv_to_fc is 'fp':
            self.fp_tensor = None
            self._build_fp()

        self._build_fc_layers()
        self._set_nonlin_and_loss()

        self.to(self.device)


    def forward(self, nn_input):
        if len(self.conv_layers):
            nn_input = self.conv_forward(nn_input)

        nn_output = self.fc_forward(nn_input)
        if self.output_fn is not None:
            nn_output = self.output_fn(nn_output)

        return nn_output


    def conv_forward(self, nn_input):
        n_pts = nn_input.size()[0]
        state_input = nn_input[:, 0:self.x_idx[-1]+1]
        image_input = nn_input[:, self.x_idx[-1]+1:self.img_idx[-1]+1]
         
        im_height = self.config['image_height']
        im_width = self.config['image_width']
        num_channels = self.config['image_channels']
        image_input = torch.view(image_input, [-1, im_width, im_height, num_channels])
        for conv_layer in self.conv_layers:
            image_input = conv_layer(image_input)
            image_input = self.act_fn(image_input)

        if self.conv_to_fc is 'fp':
            image_input = self.compute_fp(image_input)

        image_input = torch.view(image_input, [n_pts, -1])
        return torch.cat(tensors=[image_input, state_input], dim=1)


    def fc_forward(self, nn_input):
        for fc_layer in self.fc_layers[-1]:
            nn_input = fc_layer(nn_input)
            nn_input = self.act_fn(nn_input)
        return self.fc_layers[-1](nn_input)


    def _compute_idx(self):
        if 'idx' in self.config:
            self.idx = self.config['idx']
            self.x_idx, self.img_idx = [], []
            for (sensor, inds) in self.idx.items():
                if sensor in self.config['obs_image_data']:
                    self.img_idx.extend(inds)
                else:
                    self.x_idx.extend(inds)

        else:
            self.x_idx, self.img_idx, i = [], [], 0
            for sensor in self.config['obs_include']:
                dim = self.config['sensor_dims'][sensor]

                if sensor in self.config['obs_image_data']:
                    self.img_idx = self.img_idx + list(range(i, i+dim))
                else:
                    self.x_idx = self.x_idx + list(range(i, i+dim))

                i += dim


    def _set_nonlin_and_loss(self):
        self.act_fn = self.config.get('act_fn', F.relu)
        if type(self.act_fn) is str: self.act_fn = getattr(F, self.act_fn)

        self.output_fn = self.config.get('output_fn', None)
        if type(self.output_fn) is str: self.output_fn = getattr(F, self.output_fn)

        self.loss_fn = self.config.get('loss_fn', F.mse_loss)
        self.use_precision = False
        if self.loss_fn == 'precision_mse': 
            self.loss_fn = precision_mse
            self.use_precision = True
        if type(self.loss_fn) is str: self.loss_fn = getattr(F, self.loss_fn)

   
    def _build_conv_layers(self):
        num_filters = self.config.get('num_filters', [])
        filter_sizes = self.config.get('filter_sizes', [])
        self.n_conv = len(num_filters)
        if self.n_conv == 0: return

        im_height = self.config['image_height']
        im_width = self.config['image_width']
        num_channels = self.config['image_channels']
        cur_channels = num_channels

        for n in range(self.n_conv):
            conv_layer = nn.Conv2d(cur_channels, num_filters[n], filter_sizes[n])
            cur_channels = num_filters[n]
            self.conv_layers.append(conv_layer)

        # Compute size of output
        temp_model = nn.Sequential(*self.conv_layers)
        rand_inputs = torch.rand(1, num_channels, im_height, im_width)
        shape = list(temp_model(rand_inputs).shape)
        self.conv_output_dim = shape
        self.fc_input_dim = functools.reduce(operator.mul, shape)


    def _build_fc_layers(self):
        n_fc_layers = self.config.get('n_layers', 1)
        dim_hidden = self.config.get('dim_hidden', 40)
        cur_dim = self.fc_input_dim
        for n in range(n_fc_layers):
            next_dim = dim_hidden if np.isscalar(dim_hidden) else dim_hidden[n]
            fc_layer = nn.Linear(cur_dim, next_dim)
            cur_dim = next_dim
            self.fc_layers.append(fc_layer)

        fc_layer = nn.Linear(cur_dim, self.output_dim)
        self.fc_layers.append(fc_layer)


    def _build_fp(self):
        num_fp, num_rows, num_cols = self.conv_output_dim
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = torch.from_numpy(x_map)
        y_map = torch.from_numpy(y_map)

        x_map = torch.view(x_map, [num_rows * num_cols])
        y_map = torch.view(y_map, [num_rows * num_cols])
        self.fp_tensors = (x_map, y_map)

        # Compute size of output
        rand_inputs = torch.rand(1, nump_fp, num_rows, num_cols)
        shape = list(self.compute_fp(rand_inputs).shape)
        self.fc_input_dim = functools.reduce(operator.mul, shape)


    def compute_fp(self, input_layer):
        if self.fp_tensors is None: self._build_fp(input_layer)
        features = torch.view(features, [-1, num_rows*num_cols])
        softmax = torch.nn.softmax(features)
        fp_x = torch.sum(torch.multiply(x_map, softmax), dim=[1], keepdim=True)
        fp_y = torch.sum(torch.multiply(y_map, softmax), dim=[1], keepdim=True)
        fp = torch.view(torch.cat(tensors=[fp_x, fp_y], dim=1), [-1, num_fp*2])
        return fp


class PolicyNet(TorchNet):
    def __init__(self, config, scope, device=None):
        self.scope = scope
        self.normalize = self.config.get('normalize', False)
        self.scale = None
        self.bias = None

        super().__init__(config=config, device=device)


    def act(self, obs, noise=None, eta=1.):

        if self.scope in ['primitive', 'cont']:
            is_cont = self.scope == 'cont'
            return self.task_distr(obs, eta, is_cont)

        if len(np.shape(obs)) == 1:
            flatten = True
            obs = np.expand_dims(obs, axis=0)

        if self.normalize:
            if self.scale is None or self.bias is None:
                raise ValueError('scale & bias must be set before normalization')

            obs = obs.copy()
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias

        act = self.forward(obs)

        if noise is not None:
            act += noise

        return act.flatten() if flatten else act


    def task_distr(self, obs, eta=1., cont=False):
        if len(obs.shape) < 2:
            flatten = True
            obs = obs.reshape(1, -1)

        vals = self.forward(obs)

        res = []
        for bound in self._contBounds:
            res.append(vals[:, bound[0]:bound[1]])

        return res.flatten() if flatten else res


    def is_initialized(self):
        if not self.normalize: return True
        return not (self.scale is None or self.bias is None)

