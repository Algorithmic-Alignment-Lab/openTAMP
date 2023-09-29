import time
import pickle as pickle
import numpy as np
import xml.etree.ElementTree as xml

from sco_py.expr import *


import opentamp
from opentamp.envs.mjc_env import MJCEnv
import opentamp.pma.backtrack_ll_solver_OSQP as bt_ll_OSQP
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.utils.tamp_eval_funcs import *
from opentamp.policy_hooks.tamp_agent import TAMPAgent


bt_ll_OSQP.INIT_TRAJ_COEFF = 1e-2

HUMAN_TARGS = [
                (9.0, 0.),
                (9.0, 2.0),
                (9.0, 1.0),
                (9.0, -1.0),
                (9.0, -2.0),
                (9.0, -3.0),
                (9.0, -4.0),
                (-9.0, 2.),
                (-9.0, 1.),
                (-9.0, 0.),
                (-9.0, -1.0),
                (-9.0, -2.0),
                (-9.0, -3.0),
                (-9.0, -4.0),
                ]

MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100
GRIP_TOL = 0.
NEAR_TOL = 0.5
MIN_STEP = 1e-2
LIDAR_DIST = 2.
DSAFE = 5e-1
MAX_STEP = 1.0
LOCAL_FRAME = True
SPOT_XML = opentamp.__path__[0] + '/robot_info/lidar_spot_imit.xml'



class SpotAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super(SpotAgent, self).__init__(hyperparams)
        self._feasible = True

        self.check_col = hyperparams['master_config'].get('check_col', True)
        self.vel_rat = 0.05
        self.rlen = 30
        self.hor = 15
        config = {
            'obs_include': ['can{0}'.format(i) for i in range(hyperparams['num_objs'])],
            'include_files': [SPOT_XML],
            'include_items': [],
            'view': False,
            'sim_freq': 50,
            'timestep': 0.002,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height']),
            'step_mult': 5e0,
            'act_jnts': ['robot_x', 'robot_y', 'robot_theta']
        }

        self.main_camera_id = 0
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices(self.task_list)
        # for name in prim_options[OBJ_ENUM]:
        #     if name =='spot': continue
        #     cur_color = colors.pop(0)
        #     items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 40.})
        #     targ_color = cur_color[:3] + [1.]
        #     items.append({'name': '{0}_end_target'.format(name), 'type': 'box', 'is_fixed': True, 'pos': (0, 0, 1.5), 'dimensions': (0.35, 0.35, 0.045), 'rgba': tuple(targ_color), 'mass': 1.})

        # self.humans = {}
        # self.human_trajs = {}
        # for human_id in range(self.prob.N_HUMAN):
        #     self.humans['human{}'.format(human_id)] = HUMAN_TARGS[np.random.randint(len(HUMAN_TARGS))]
        #     self.human_trajs['human{}'.format(human_id)] = np.zeros(2) 
        #     items.append({'name': 'human{}'.format(human_id),
        #                   'type': 'cylinder',
        #                   'is_fixed': False,
        #                   'pos': [0., 0., 0.],
        #                   'dimensions': [0.3, 0.2],
        #                   'mass': 10,
        #                   'rgba': (1., 1., 1., 1.)})

        no = self._hyperparams['num_objs']
        nt = self._hyperparams['num_targs']
        config['load_render'] = hyperparams['master_config'].get('load_render', False)
        config['xmlid'] = '{0}_{1}_{2}_{3}'.format(self.process_id, self.rank, no, nt)
        self.mjc_env = MJCEnv.load_config(config)
        self.targ_labels = {i: np.array(self.prob.END_TARGETS[i]) for i in range(len(self.prob.END_TARGETS))}
        if hasattr(self.prob, 'ALT_END_TARGETS'):
            self.alt_targ_labels = {i: np.array(self.prob.ALT_END_TARGETS[i]) for i in range(len(self.prob.ALT_END_TARGETS))}
        else:
            self.alt_targ_labels = self.targ_labels
        self.targ_labels.update({i: self.targets[0]['aux_target_{0}'.format(i-no)] for i in range(no, no+self.prob.n_aux)})


    def run_policy_step(self, u, x):
        self.mjc_env.step(u)

        if not self._feasible:
            return False, 0

        for human in self.humans:
            if not self._eval_mode:
                self.mjc_env.physics.named.data.qvel[human][:] = 0.
            else:
                self.mjc_env.physics.named.data.qvel[human][:2] = self.human_trajs[human]
                self.mjc_env.physics.named.data.qvel[human][2:] = 0. 
        self.mjc_env.physics.forward()

        self._col = []

        cmd_theta = u[self.action_inds['spot', 'theta']][0]
        if ('spot', 'pose') not in self.action_inds:
            cmd_vel = u[self.action_inds['spot', 'vel']]
            self.mjc_env.set_user_data('vel', cmd_vel)
            cur_theta = x[self.state_inds['spot', 'theta']][0]
            cmd_x, cmd_y = -cmd_vel*np.sin(cur_theta), cmd_vel*np.cos(cur_theta)
        else:
            cur_theta = x[self.state_inds['spot', 'theta']][0]
            rel_x, rel_y = u[self.action_inds['spot', 'pose']]
            cmd_x, cmd_y = rel_x, rel_y
            if LOCAL_FRAME:
                cmd_x, cmd_y = np.array([[np.cos(cur_theta), -np.sin(cur_theta)], \
                                         [np.sin(cur_theta), np.cos(cur_theta)]]).dot([rel_x, rel_y])

        if np.isnan(cmd_x): cmd_x = 0
        if np.isnan(cmd_y): cmd_y = 0
        if np.isnan(cmd_theta): cmd_theta = 0

        nsteps = int(max(abs(cmd_x), abs(cmd_y)) / self.vel_rat) + 1
        cur_x, cur_y, _ = self.mjc_env.get_item_pos('spot')
        ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta])
        for n in range(nsteps):
            x = cur_x + float(n)/nsteps * cmd_x
            y = cur_y + float(n)/nsteps * cmd_y
            theta = cur_theta + float(n)/nsteps * cmd_theta
            ctrl_vec = np.array([x, y, theta])
            self.mjc_env.step(ctrl_vec, mode='velocity', gen_obs=False)

        ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta])
        self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')

        for human in self.humans:
            if np.linalg.norm(self.mjc_env.get_item_pos(human)[:2] - self.mjc_env.get_item_pos('spot')[:2]) < 0.65:
                self._feasible = False

        col = 1 if len(self._col) > 0 else 0
        self._rew = self.reward()
        self._ret += self._rew
        return True, col


    def get_state(self):
        x = np.zeros(self.dX)
        for pname, attr in self.state_inds:
            if attr == 'pose':
                val = self.mjc_env.get_item_pos(pname)
                x[self.state_inds[pname, attr]] = val[:2]
            elif attr == 'rotation':
                val = self.mjc_env.get_item_rot(pname)
                x[self.state_inds[pname, attr]] = val
            elif attr == 'robot_theta':
                val = self.mjc_env.get_joints(['robot_theta'])
                val = val['robot_theta']
                x[self.state_inds[pname, 'robot_theta']] = val
            elif attr == 'robot_x':
                val = self.mjc_env.get_joints(['robot_x'])
                val = val['robot_x']
                x[self.state_inds[pname, 'robot_x']] = val
            elif attr == 'robot_y':
                val = self.mjc_env.get_joints(['robot_y'])
                val = val['robot_y']
                x[self.state_inds[pname, 'robot_y']] = val
            elif attr == 'vel':
                val = self.mjc_env.get_user_data('vel', 0.)
                x[self.state_inds[pname, 'vel']] = val

        assert not np.any(np.isnan(x))
        return x.round(5)


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        if task is None:
            task = list(self.plans.keys())[0]
        mp_state = mp_state.copy()
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task]
        ee_pose = mp_state[self.state_inds['spot', 'pose']]
        if targets is None:
            targets = self.target_vecs[cond].copy()

        theta = mp_state[self.state_inds['spot', 'theta']][0]
        while theta < -np.pi:
            theta += 2*np.pi
        while theta > np.pi:
            theta -= 2*np.pi

        if LOCAL_FRAME:
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
        else:
            rot = np.eye(2)

        sample.set(EE_ENUM, ee_pose, t)
        sample.set(THETA_ENUM, np.array([theta]), t)
        dirvec = np.array([-np.sin(theta), np.cos(theta)])
        sample.set(THETA_VEC_ENUM, dirvec, t)
        velx = self.mjc_env.physics.named.data.qvel['robot_x'][0]
        vely = self.mjc_env.physics.named.data.qvel['robot_y'][0]
        sample.set(VEL_ENUM, np.array([velx, vely]), t)
        sample.set(STATE_ENUM, mp_state, t)

        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            for human in self.humans:
                self._x_delta[:,self.state_inds[human, 'pose']] = 0.
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)
            sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)

        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)
        
        prim_choices = self.prob.get_prim_choices(self.task_list)

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.task_ind = task[0]
        sample.set(TASK_ENUM, task_vec, t)

        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        grasp = np.array([0, -0.601])
        onehottask = tuple([val for val in task if np.isscalar(val)])
        sample.set(FACTOREDTASK_ENUM, np.array(onehottask), t)

        if OBJ_ENUM in prim_choices:
            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            if self.task_list[task[0]].find('place') >= 0:
                obj_vec[:] = 1. / len(obj_vec)

            sample.obj_ind = task[1]
            obj_ind = task[1]
            obj_name = list(prim_choices[OBJ_ENUM])[obj_ind]
            sample.set(OBJ_ENUM, obj_vec, t)
            obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[self.state_inds['spot', 'pose']]
            base_pos = obj_pose
            obj_pose = rot.dot(obj_pose)
            sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
            sample.obj = task[1]
            if self.task_list[task[0]].find('move') >= 0:
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds[obj_name, 'pose']].copy(), t)

        if TARG_ENUM in prim_choices:
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            targ_vec[task[2]] = 1.
            if self.task_list[task[0]].find('move') >= 0:
                targ_vec[:] = 1. / len(targ_vec)
            sample.targ_ind = task[2]
            targ_ind = task[2]
            targ_name = list(prim_choices[TARG_ENUM])[targ_ind]
            sample.set(TARG_ENUM, targ_vec, t)
            targ_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds['spot', 'pose']]
            targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds[obj_name, 'pose']]
            base_pos = targ_pose
            targ_pose = rot.dot(targ_pose)
            targ_off_pose = rot.dot(targ_off_pose)
            sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)
            sample.targ = task[2]
            if self.task_list[task[0]].find('place') >= 0 or self.task_list[task[0]].find('transfer') >= 0:
                sample.set(END_POSE_ENUM, targ_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, targets[self.target_inds[targ_name, 'value']].copy(), t)

        sample.set(TRUE_POSE_ENUM, sample.get(ABS_POSE_ENUM, t=t), t)
        if ABS_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(ABS_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(ABS_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind] - mp_state[self.state_inds['spot', 'pose']]), t)

        if REL_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(REL_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(REL_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind]), t)

        if END_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(END_POSE_ENUM)
            if ind < len(task) and type(task[ind]) is not int:
                sample.set(END_POSE_ENUM, task[ind], t)

        sample.task = task
        sample.condition = cond
        sample.task_name = self.task_list[task[0]]
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(GOAL_ENUM, np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]]), t)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
            sample.set(ONEHOT_GOAL_ENUM, self.onehot_encode_goal(sample.get(GOAL_ENUM, t)), t)
        sample.targets = targets.copy()

        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            #sample.set(OBJ_DELTA_ENUMS[i], rot.dot(mp_state[self.state_inds[obj, 'pose']]-ee_pose), t)
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose, t)
            sample.set(TARG_ENUMS[i], targ, t)
            #sample.set(TARG_DELTA_ENUMS[i], rot.dot(targ-mp_state[self.state_inds[obj, 'pose']]), t)
            sample.set(TARG_DELTA_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

        if INGRASP_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - mp_state[self.state_inds['spot', 'pose']] - grasp) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(INGRASP_ENUM, vec, t=t)

        if ATGOAL_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - targets[self.target_inds['{0}_end_target'.format(o), 'value']]) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(ATGOAL_ENUM, vec, t=t)

        if fill_obs:
            if LIDAR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                set_params_attrs(plan.params, plan.state_inds, mp_state, 0)
                lidar = self.dist_obs(plan, 1)
                sample.set(LIDAR_ENUM, lidar.flatten(), t)

            if MJC_SENSOR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                sample.set(MJC_SENSOR_ENUM, self.mjc_env.get_sensors(), t)

            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include'] or \
               IM_ENUM in self._hyperparams['cont_obs_include']:
                im = self.mjc_env.render(height=self.image_height, width=self.image_width, view=self.view)
                im = (im - 128.) / 128.
                sample.set(IM_ENUM, im.flatten().astype(np.float32), t)


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._ret = 0.
        self._rew = 0.
        self._feasible = True
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._x_delta[:] = x.reshape((1,-1))
        self.eta_scale = 1.
        self._noops = 0
        self.mjc_env.reset()
        xval, yval = mp_state[self.state_inds['spot', 'pose']]
        theta = x[self.state_inds['spot', 'theta']][0]
        self.mjc_env.set_user_data('vel', 0.)
        self.mjc_env.set_joints({'robot_x': xval, 'robot_y': yval, 'robot_theta': theta}, forward=False)
        for param_name, attr in self.state_inds:
            if param_name == 'spot': continue
            if attr == 'pose':
                pos = mp_state[self.state_inds[param_name, 'pose']].copy()
                self.mjc_env.set_item_pos(param_name, np.r_[pos, 0.5], forward=False)
                if param_name.find('can') >= 0:
                    targ = self.target_vecs[0][self.target_inds['{0}_end_target'.format(param_name), 'value']]
                    self.mjc_env.set_item_pos('{0}_end_target'.format(param_name), np.r_[targ, -0.15], forward=False)
        self.mjc_env.physics.data.qvel[:] = 0.
        self.mjc_env.physics.forward()


    def set_to_targets(self, condition=0):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        objs = prim_choices[OBJ_ENUM]
        for obj_name in objs:
            self.mjc_env.set_item_pos(obj_name, np.r_[self.targets[condition]['{0}_end_target'.format(obj_name)], 0], forward=False)
        self.mjc_env.physics.forward()


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        act_st, et = plan.actions[anum].active_timesteps
        st = max(act_st, st)
        if targets is None:
            targets = self.target_vecs[cond].copy()
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]] == 'moveto':
            params[3].value[:,0] = params[0].pose[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,st]


    def replace_cond(self, cond, curric_step=-1):
        self.prob.NUM_OBJS = self.config['num_objs']
        self.prob.NUM_TARGS = self.config['num_objs']
        self.prob.n_aux = 0
        self.init_vecs[cond], self.targets[cond] = self.prob.get_random_initial_state_vec(self.config, self._eval_mode, self.dX, self.state_inds, 1)
        self.init_vecs[cond], self.targets[cond] = self.init_vecs[cond][0], self.targets[cond][0]
        if self.swap:
            objs = self.prim_choices[OBJ_ENUM]
            inds = np.random.permutation(len(objs))
            for i, ind in enumerate(inds):
                if i == ind: continue
                pos1_inds = self.state_inds[objs[i], 'pose']
                targ = '{}_end_target'.format(objs[ind])
                pos2_inds = self.target_inds[targ, 'value']
                noise = np.random.normal(0, 0.1, len(pos2_inds))
                self.init_vecs[cond][pos1_inds] = self.targets[cond][targ] + noise

        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        # print(self.targets)
        # print(self.target_vecs)
        # for target_name in self.targets[cond]:
        #     self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]



    def goal(self, cond, targets=None):
        if self.goal_type == 'moveto':
            assert ('can1', 'pose') not in self.state_inds
            return '(NearGraspAngle  spot can0) '
        if targets is None:
            targets = self.target_vecs[cond]
        prim_choices = self.prob.get_prim_choices(self.task_list)
        goal = ''
        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            targ_labels = self.targ_labels if not self._eval_mode else self.alt_targ_labels
            for ind in targ_labels:
                if np.all(np.abs(targ - targ_labels[ind]) < NEAR_TOL):
                    goal += '(Near {0} end_target_{1}) '.format(obj, ind)
                    break
        return goal

    def goal_f(self, condition, state, targets=None, cont=False):
        return 1.0 # TEMPLATE


    def get_annotated_image(self, s, t, cam_id=None):
        if cam_id is None: cam_id = self.camera_id
        x = s.get_X(t=t)
        task = s.get(FACTOREDTASK_ENUM, t=t).astype(int)
        pos = s.get(TRUE_POSE_ENUM, t=t)
        predpos = s.get(ABS_POSE_ENUM, t=t)
        precost = round(self.precond_cost(s, tuple(task), t), 5)
        postcost = round(self.postcond_cost(s, tuple(task), t, x0=s.base_x), 5)
        offset = str((pos - predpos).round(2))[1:-1]
        act = str(s.get(ACTION_ENUM, t=t).round(3))[1:-1]
        textover1 = self.mjc_env.get_text_overlay(body='Task: {0} {1}'.format(task, act))
        textover2 = self.mjc_env.get_text_overlay(body='{0: <6} {1: <6} Error: {2}'.format(precost, postcost, offset), position='bottom left')
        self.reset_to_state(x)
        im = self.mjc_env.render(camera_id=cam_id, height=self.image_height, width=self.image_width, view=False, overlays=(textover1, textover2))
        return im


    def feasible_state(self, x, targets):
        return self._feasible


    def human_cost(self, x, goal_wt=1e1, col_wt=2e0, rcol_wt=5e0):
        cost = 0
        for human in self.humans:
            hpos = x[self.state_inds[human, 'pose']]
            cost += goal_wt * np.linalg.norm(hpos-self.humans[human])

            for (pname, aname), inds in self.state_inds.items():
                if pname == human: continue
                if aname != 'pose': continue
                if pname.find('spot') >= 0 and np.linalg.norm(x[inds]-hpos) < 0.8:
                    cost += rcol_wt
                elif pname.find('human') >= 0 and np.linalg.norm(x[inds]-hpos) < 0.8:
                    cost += rcol_wt
                elif pname.find('can') >= 0:
                    cost -= col_wt * np.linalg.norm(x[inds]-hpos)
        return cost


    def solve_humans(self, policy, task, hor=2, N=30):
        if not self._eval_mode or not self.prob.N_HUMAN:
            for n in range(self.prob.N_HUMAN):
                self.human_trajs['human{}'.format(n)] = np.zeros(2)
            return

        for human_id in range(self.prob.N_HUMAN):
            if np.random.uniform() < 0.05:
                self.humans['human{}'.format(human_id)] = HUMAN_TARGS[np.random.randint(len(HUMAN_TARGS))]

        old_feas = self._feasible
        self._feasible = True
        init_t = time.time()
        qpos = self.mjc_env.physics.data.qpos.copy()
        qvel = self.mjc_env.physics.data.qvel.copy()
        init_state = self.get_state()
        trajs = []
        sample = Sample(self)
        for _ in range(N):
            self.mjc_env.physics.data.qpos[:] = qpos.copy()
            self.mjc_env.physics.data.qvel[:] = qvel.copy()
            self.mjc_env.physics.forward()
            #traj = np.random.uniform(-2, 2, (self.prob.N_HUMAN, hor, 2))
            traj = np.random.uniform(-1, 1, (self.prob.N_HUMAN, hor, 2))
            traj[:,:,1] *= 2
            cost = 0
            for t in range(hor):
                x = self.get_state()
                self.fill_sample(0, sample, x, t, task, fill_obs=True)
                act = policy.act(sample.get_X(t=t), sample.get_obs(t=t), t)
                for n in range(self.prob.N_HUMAN):
                    self.human_trajs['human{}'.format(n)] = traj[n, t]
                self.run_policy_step(act, x)
                self._feasible = True
                goal_wt = 0 if t < hor-1 else hor * 1e0
                cost += self.human_cost(x, goal_wt=goal_wt)
            trajs.append((cost, traj[:,0]))

        self.mjc_env.physics.data.qpos[:] = qpos
        self.mjc_env.physics.data.qvel[:] = qvel
        self.mjc_env.physics.forward()
        cur_cost, cur_traj = trajs[0]
        for cost, traj in trajs:
            if cost < cur_cost:
                cur_cost = cost
                cur_traj = traj

        for n in range(self.prob.N_HUMAN):
            self.human_trajs['human{}'.format(n)] = traj[n]

        #print('TIME TO GET HUMAN ACTS FOR {} N {} HOR: {}'.format(N, hor, time.time() - init_t))
        self._feasible = old_feas
        return traj


    def reward(self, x=None, targets=None, center=False, gamma=0.9):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        l2_coeff = 2e-2 # 1e-2
        log_coeff = 1.
        obj_coeff = 1.
        targ_coeff = 1.

        opts = self.prob.get_prim_choices(self.task_list)
        rew = 0
        eeinds = self.state_inds['spot', 'pose']
        ee_pos = x[eeinds]
        ee_theta = x[self.state_inds['spot', 'theta']][0]
        dist = 0.61
        tol_coeff = 0.8
        grip_pos = ee_pos + [-dist*np.sin(ee_theta), dist*np.cos(ee_theta)]
        max_per_obj = 3.2
        info_per_obj = []
        min_dist = np.inf
        for opt in opts[OBJ_ENUM]:
            xinds = self.state_inds[opt, 'pose']
            targinds = self.target_inds['{}_end_target'.format(opt), 'value']
            dist_to_targ = np.linalg.norm(x[xinds]-targets[targinds])
            dist_to_grip = np.linalg.norm(grip_pos - x[xinds])

            if dist_to_targ < tol_coeff*NEAR_TOL:
                rew += 2 * (obj_coeff + targ_coeff) * max_per_obj / (1-gamma)
                info_per_obj.append((np.inf,0))
            else:
                grip_l2_term = -l2_coeff * dist_to_grip**2
                grip_log_term = -np.log(log_coeff * dist_to_grip + 1e-6)
                targ_l2_term = -l2_coeff * dist_to_targ**2
                targ_log_term = -log_coeff * np.log(dist_to_targ + 1e-6)
                grip_obj_rew = obj_coeff * np.min([grip_l2_term + grip_log_term, max_per_obj])
                targ_obj_rew = targ_coeff * np.min([targ_l2_term + targ_log_term, max_per_obj])
                rew += targ_obj_rew # Always penalize obj to target distance
                min_dist = np.min([min_dist, dist_to_grip])
                info_per_obj.append((dist_to_grip, grip_obj_rew)) # Only penalize closest object to gripper

        for dist, obj_rew in info_per_obj:
            if dist <= min_dist:
                rew += obj_rew
                break

        return rew / 1e1


    def permute_tasks(self, tasks, targets, plan=None, x=None):
        encoded = [list(l) for l in tasks]
        no = self._hyperparams['num_objs']
        perm = np.random.permutation(range(no))
        for l in encoded:
            l[1] = perm[l[1]]
        encoded = [tuple(l) for l in encoded]
        target_vec = targets.copy()
        param_map = {}
        old_values = {}
        for n in range(no):
            inds = self.target_inds['can{0}_end_target'.format(n), 'value']
            inds2 = self.target_inds['can{0}_end_target'.format(perm[n]), 'value']
            target_vec[inds2] = targets[inds]
            if plan is None:
                can = 'can{0}'.format(n)
                old_values[can] = x[self.state_inds[can, 'pose']]
            else:
                old_values['can{0}'.format(n)] = plan.params['can{0}'.format(n)].pose.copy()
        perm_map = {}
        for n in range(no):
            perm_map['can{0}'.format(n)] = 'can{0}'.format(perm[n])
        return encoded, target_vec, perm_map


    def encode_plan(self, plan, permute=False):
        encoded = []
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for a in plan.actions:
            encoded.append(self.encode_action(a))

        for i, l in enumerate(encoded[:-1]):
            if self.task_list[l[0]].find( 'moveto') >= 0 and self.task_list[encoded[i+1][0]].find('transfer') >= 0:
                l[2] = encoded[i+1][2]
        encoded = [tuple(l) for l in encoded]
        return encoded


    def encode_action(self, action):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        keys = list(prim_choices.keys())
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower().find(task) >= 0:
                l[0] = i
                break

        for enum in prim_choices:
            if enum is TASK_ENUM or np.isscalar(prim_choices[enum]): continue
            l.append(0)
            for i, opt in enumerate(prim_choices[enum]):
                if opt.lower() in [p.name.lower() for p in action.params][:-1]:
                    l[-1] = i
                    break

        if self.task_list[l[0]].find('move') >= 0 and TARG_ENUM in prim_choices:
            l[keys.index(TARG_ENUM)] = np.random.randint(len(prim_choices[TARG_ENUM]))
        return l
    