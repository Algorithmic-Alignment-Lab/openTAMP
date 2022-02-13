""" This file provides an example tensorflow network used to define a policy. """

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def precision_mse(output, y, precision):
    return torch.matmul(torch.matmul(output, precision), y.transpose())


class PolicyNet(nn.Module):
    def __init__(self, config, device=None):
        self.config = config

        if device is None: device = torch.device('cpu')
        if type(device) is str: device = torch.device(device)
        self.device = device

        self.conv_layers = []
        self.fc_layers = []

        self._compute_idx()
        self._build_conv_layers()
        self._build_fc_layers()
        self._set_nonlin_and_loss()
        
        self.conv_to_fc = config.get('conv_to_fc', 'fp')
        if self.conv_to_fc is 'fp':
            self.fp_tensor = None
            self._build_fp()

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


    def _compute_idx(self, config):
        if 'idx' in config:
            self.x_idx, self.img_idx = config['idx']
        else:
            x_idx, img_idx, i = [], [], 0
            for sensor in config['obs_include']:
                dim = config['sensor_dims'][sensor]
                if sensor in config['obs_image_data']:
                    img_idx = img_idx + list(range(i, i+dim))
                else:
                    x_idx = x_idx + list(range(i, i+dim))
                i += dim
            self.x_idx = x_idx
            self.img_idx = img_idx


    def _set_nonlin_and_loss(self, config):
        self.act_fn = config.get('act_fn', F.relu)
        if type(self.act_fn) is str: self.act_fn = getattr(F, self.act_fn)

        self.output_fn = config.get('output_fn', None)
        if type(self.output_fn) is str: self.output_fn = getattr(F, self.output_fn)

        self.loss_fn = config.get('loss_fn', F.mse_loss)
        if self.loss_fn == 'precision_mse': self.loss_fn = precision_mse
        if type(self.loss_fn) is str: self.loss_fn = getattr(F, self.loss_fn)

   
    def _build_conv_layers(self, config):
        num_filters = config.get('num_filters', [])
        filter_sizes = config.get('filter_sizes', [])
        self.n_conv = len(num_filters)
        im_height = config['image_height']
        im_width = config['image_width']
        num_channels = config['image_channels']

        cur_channels = num_channels
        for n in range(self.n_conv):
            conv_layer = nn.Conv2d(cur_channels, num_filters[n], filter_sizes[n])
            cur_channels = num_channels[n]
            self.conv_layers.append(conv_layer)


    def _build_fc_layers(self, dim_input, dim_output, config):
        n_fc_layers = config.get('n_layers', 1)
        dim_hidden = config.get('dim_hidden', 40)
        cur_dim = dim_input
        for n in range(n_fc_layers):
            next_dim = dim_hidden if np.isscalar(dim_hidden) else dim_hidden[n]
            fc_layer = nn.Linear(cur_dim, next_dim)
            cur_dim = next_dim
            self.fc_layers.append(fc_layer)

        fc_layer = nn.Linear(cur_dim, dim_output)
        self.fc_layers.append(fc_layer)


    def _build_fp(self, input_layer):
        _, num_rows, num_cols, num_fp = input_layer.size()
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

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


    def _compute_fp(self, input_layer):
        if self.fp_tensors is None: self._build_fp(input_layer)
        features = torch.transpose(input_layer, [0,3,1,2])
        features = torch.view(features, [-1, num_rows*num_cols])
        softmax = torch.nn.softmax(features)
        fp_x = torch.sum(torch.multiply(x_map, softmax), dim=[1], keepdim=True)
        fp_y = torch.sum(torch.multiply(y_map, softmax), dim=[1], keepdim=True)
        fp = torch.view(torch.cat(tensors=[fp_x, fp_y], dim=1), [-1, num_fp*2])
        return fp

