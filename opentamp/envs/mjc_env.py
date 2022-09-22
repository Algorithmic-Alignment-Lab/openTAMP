import matplotlib.pyplot as plt
import numpy as np
import os
import random
from threading import Thread
import time
import traceback
import sys
import xml.etree.ElementTree as xml

from dm_control.mujoco import Physics, TextOverlay
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.rl.control import PhysicsError

from gym import spaces
from gym.core import Env

import opentamp
from opentamp.envs.mjc_xml_utils import *
from opentamp.envs import transform_utils as T


BASE_XML = opentamp.__path__._path[0] +'/robot_info/empty.xml'
ENV_XML = opentamp.__path__._path[0] + '/robot_info/current_empty.xml'
SPECIFIC_ENV_XML = opentamp.__path__._path[0] + '/robot_info/temp_env_xmls/current_{0}.xml'

_MAX_FRONTBUFFER_SIZE = 2048

CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos', 'discrete']

class MJCEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    def __init__(self, mode='end_effector', obs_include=[], items=[], include_files=[], include_items=[], im_dims=(DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), sim_freq=25, timestep=0.002, max_iter=250, mult=3e2, view=False, load_render=True, act_jnts=[], xmlid='0'):
        assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
        self.ctrl_mode = mode
        self.active = True

        self.cur_time = 0.
        self.prev_time = 0.
        self.timestep = timestep
        self.sim_freq = sim_freq
        self.mult = 3e2

        self.use_viewer = view
        self.use_glew = 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] == 'glfw'
        self.obs_include = obs_include
        self._joint_map_cache = {}
        self._ind_cache = {}
        self._type_cache = {}
        self._user_data = {}
        self._cache_rendering = False 
        self._cached_images = {}
        self._last_rendered_state = (None, None)

        self.im_wid, self.im_height = im_dims
        self.items = items
        self._item_map = {item[0]: item for item in items}
        self.include_files = include_files
        self.include_items = include_items
        self.item_names = list(self._item_map.keys()) + [item['name'] for item in include_items]
        self.act_jnts = act_jnts
        self.xmlid = xmlid

        self._load_model()
        self._set_obs_info(obs_include)
        for item in self.include_items:
            if item.get('is_fixed', False): continue
            name = item['name']
            pos = item.get('pos', (0, 0, 0))
            quat = item.get("quat", (1, 0, 0, 0))

            self.set_item_pos(name, pos)
            self.set_item_rot(name, quat)
        self.init_state = self.physics.data.qpos.copy()

        self._init_control_info()

        self._max_iter = max_iter
        self._cur_iter = 0

        self.load_render = load_render
        if self.load_render:
            try:
                from dm_control import render
            except:
                from dm_control import _render as render

        self._viewer = None
        if view and self.load_render:
            self.add_viewer()
        
        self.render(camera_id=0)
        self.render(camera_id=0)


    @classmethod
    def load_config(cls, config):
        mode = config.get("mode", "joint_angle")
        obs_include = config.get("obs_include", [])
        items = config.get("items", [])
        include_files = config.get("include_files", [])
        include_items = config.get("include_items", [])
        im_dims = config.get("image_dimensions", (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT))
        sim_freq = config.get("sim_freq", 25)
        ts = config.get("mjc_timestep", 0.002)
        mult = config.get("step_mult", 3e2)
        view = config.get("view", False)
        max_iter = config.get("max_iterations", 250)
        load_render = config.get("load_render", True)
        act_jnts = config.get("act_jnts", [])
        xmlid = config.get("xmlid", 0)
        return cls(mode, obs_include, items, include_files, include_items, im_dims, sim_freq, ts, max_iter, mult, view, load_render=load_render, act_jnts=act_jnts, xmlid=xmlid)


    def _load_model(self):
        xmlpath = SPECIFIC_ENV_XML.format(self.xmlid)
        generate_xml(BASE_XML, xmlpath, self.items, self.include_files, self.include_items, timestep=self.timestep, frame_dims=(self.im_wid, self.im_height))
        self.physics = Physics.from_xml_path(xmlpath)


    def _init_control_info(self):
        print('No control information to initialize.')


    def add_viewer(self):
        if self._viewer is not None: return
        self.cur_im = np.zeros((self.im_height, self.im_wid, 3))
        self._launch_viewer(self.im_wid, self.im_height)


    def _launch_viewer(self, width, height, title='Main'):
        self._matplot_view_thread = None
        if self.use_glew:
            from dm_control.viewer import viewer
            from dm_control.viewer import views
            from dm_control.viewer import gui
            from dm_control.viewer import renderer
            self._renderer = renderer.NullRenderer()
            self._render_surface = None
            self._viewport = renderer.Viewport(width, height)
            self._window = gui.RenderWindow(width, height, title)
            self._viewer = viewer.Viewer(
                self._viewport, self._window.mouse, self._window.keyboard)
            self._viewer_layout = views.ViewportLayout()
            self._viewer.render()
        else:
            self._viewer = None
            self._matplot_im = None
            self._run_matplot_view()


    def _reload_viewer(self):
        if self._viewer is None or not self.use_glew: return

        if self._render_surface:
          self._render_surface.free()

        if self._renderer:
          self._renderer.release()

        self._render_surface = render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._viewer.zoom_to_scene()


    def _render_viewer(self, pixels):
        if self.use_glew:
            with self._window._context.make_current() as ctx:
                ctx.call(
                    self._window._update_gui_on_render_thread, self._window._context.window, pixels)
            self._window._mouse.process_events()
            self._window._keyboard.process_events()
        else:
            if self._matplot_im is not None:
                self._matplot_im.set_data(pixels)
                plt.draw()


    def _run_matplot_view(self):
        self._matplot_view_thread = Thread(target=self._launch_matplot_view)
        self._matplot_view_thread.daemon = True
        self._matplot_view_thread.start()


    def _launch_matplot_view(self):
        try:
            # self._matplot_im = plt.imshow(self.render(view=False))
            self._matplot_im = plt.imshow(self.cur_im)
            plt.show()
        except Error:
            print('\nCould not find display to launch viewer (this does not affect the ability to render images)\n')


    @property
    def qpos(self):
        return self.physics.data.qpos


    @property
    def qvel(self):
        return self.physics.data.qvel


    @property
    def qacc(self):
        return self.physics.data.qacc


    def step(self, action, mode=None, obs_include=None, gen_obs=True, view=False, debug=False):
        for t in range(self.sim_freq):
            cur_state = self.physics.data.qpos.copy()
            cur_act = self.get_jnt_vec(self.act_jnts)
            if mode is None or mode == 'position' or mode == 'joint_angle':
                self.physics.set_control(action)
            elif mode == 'velocity':
                self.physics.set_control(self.mult*(action-cur_act))
               
            qacc = self.physics.data.actuator_force.copy()
            try:
                self.physics.step()
            except PhysicsError as e:
                #traceback.print_exception(*sys.exc_info())
                print('\nERROR IN PHYSICS SIMULATION; RESETTING ENV.\n')
                self.physics.reset()
                self.physics.data.qpos[:] = cur_state[:]
                self.physics.forward()

        if not gen_obs: return
        return self.get_obs(obs_include=obs_include, view=view), \
               self.compute_reward(), \
               self.is_done(), \
               {}

    def get_sensors(self, sensors=[]):
        if not len(sensors):
            return self.physics.data.sensordata.copy()
        inds = [self.physics.model.name2id[s] for s in sensors]
        return self.physics.data.sensordata[inds]


    def get_state(self):
        return self.physics.data.qpos.copy()


    def set_state(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()


    '''
    def __getstate__(self):
        return self.physics.data.qpos.tolist()
    '''

    '''
    def __setstate__(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()
    '''


    def _set_obs_info(self, obs_include):
        self._obs_inds = {}
        self._obs_shape = {}
        ind = 0
        if 'overhead_image' in obs_include or not len(obs_include):
            self._obs_inds['overhead_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['overhead_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        # if 'forward_image' in obs_include or not len(obs_include):
        #     self._obs_inds['forward_image'] = (ind, ind+3*self.im_wid*self.im_height)
        #     self._obs_shape['forward_image'] = (self.im_height, self.im_wid, 3)
        #     ind += 3*self.im_wid*self.im_height

        for item, xml, info in self.items:
            if item in obs_include or not len(obs_include):
                self._obs_inds[item] = (ind, ind+3) # Only store 3d Position
                self._obs_shape[item] = (3,)
                ind += 3

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None, view=False):
        obs = np.zeros(self.dO)
        if obs_include is None:
            obs_include = self.obs_include

        if self.load_render:
            if view or not len(obs_include) or 'overhead_image' in obs_include:
                pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=0, view=view)
                if 'overhead_image' in self._obs_inds:
                    inds = self._obs_inds['overhead_image']
                    obs[inds[0]:inds[1]] = pixels.flatten()

            # if not len(obs_include) or 'forward_image' in obs_include:
            #     pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=1, view=view)
            #     inds = self._obs_inds['forward_image']
            #     obs[inds[0]:inds[1]] = pixels.flatten()

        for item in self.items:
            if not len(obs_include) or item[0] in obs_include:
                inds = self._obs_inds[item[0]]
                obs[inds[0]:inds[1]] = self.get_item_pos(item[0])

        return np.array(obs)


    def get_obs_types(self):
        return list(self._obs_inds.keys())


    def get_obs_inds(self, obs_type):
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        return self._obs_inds[obs_type]


    def get_obs_shape(self, obs_type):
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        return self._obs_shape[obs_type]


    def get_obs_data(self, obs, obs_type):
        obs = np.array(obs)
        if obs_type not in self._obs_inds:
            raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
        inds = self._obs_inds[obs_type]
        return obs[inds[0]:inds[1]].reshape(self._obs_shape[obs_type])


    def get_attr(self, name, attr, mujoco_frame=True):
        if attr.find('ee_pos') >= 0:
            name = attr.replace('ee_pos', 'gripper')
            attr = 'pose'
        
        if attr in self.geom.jnt_names:
            jnts = self._jnt_inds[attr]
            bnds = self.geom.get_joint_limits(attr)
            vals = self.get_joints(jnts, vec=True)
            return np.maximum(np.minimum(bnds[1], vals), bnds[0])
        
        if attr == 'pose' or attr == 'pos':
            return self.get_item_pos(name, mujoco_frame)

        if attr in ['rot', 'rotation', 'quat', 'euler']:
            euler = attr == 'euler'
            return self.get_item_rot(name, mujoco_frame, euler)
        
        if hasattr(self, 'get_{}'.format(attr)):
            return getattr(self, 'get_{}'.format(attr))(name, mujoco_frame=True)

        raise NotImplementedError('Could not retrieve value of {} for {}'.format(attr, name))


    def set_attr(self, name, attr, val, mujoco_frame=True, forward=True):
        if attr in self.geom.jnt_names:
            jnts = self.geom.jnt_names[attr]
            if len(val) == 1: val = [val[0] for _ in jnts]
            return self.set_joints(dict(zip(jnts, val)), forward=forward)
        
        if attr == 'pose' or attr == 'pos':
            return self.set_item_pos(name, val, mujoco_frame, forward=forward)

        if attr in ['rot', 'rotation', 'quat', 'euler']:
            return self.set_item_rot(name, val, mujoco_frame, forward=forward)
        
        if hasattr(self, 'set_{}'.format(attr)):
            return getattr(self, 'set_{}'.format(attr))(name, val, mujoco_frame, forward=forward)

        raise NotImplementedError('Could not set value of {} for {}'.format(attr, name))


    def get_pos_from_label(self, label, mujoco_frame=True):
        try:
            pos = self.get_item_pos(label, mujoco_frame)
        except:
            pos = None

        return pos


    def get_item_pos(self, name, mujoco_frame=True, rot=False):
        model = self.physics.model
        item_type = 'joint'
        if name in self._type_cache:
            item_type = self._type_cache[name]

        pos = [np.nan, np.nan, np.nan]
        if rot: pos.append(np.nan)
        if item_type == 'joint':
            try:
                ind = model.name2id(name, 'joint')
                adr = model.jnt_qposadr[ind]
                if rot:
                    pos = self.physics.data.qpos[adr+3:adr+7].copy()
                else:
                    pos = self.physics.data.qpos[adr:adr+3].copy()
                self._type_cache[name] = 'joint'
            except Exception as e:
                item_type = 'body'
        if item_type == 'body':
            try:
                item_ind = model.name2id(name, 'body')
                arr = self.physics.data.xquat if rot else self.physics.data.xpos
                pos = arr[item_ind].copy()
                # pos = self.physics.data.xpos[item_ind].copy()
                self._type_cache[name] = 'body'
            except Exception as e:
                item_ind = -1
    
        assert not np.any(np.isnan(pos))
        return pos


    def get_item_rot(self, name, mujoco_frame=True, to_euler=False):
        rot = self.get_item_pos(name, mujoco_frame, True)
        if to_euler:
            rot = T.quaternion_to_euler(rot)

        return rot


    def set_item_pos(self, name, pos, mujoco_frame=True, forward=True, rot=False):
        item_type = 'joint'
        if np.any(np.isnan(pos)): return
        if name in self._type_cache:
            item_type = self._type_cache[name]

        if item_type == 'joint':
            try:
                ind = self.physics.model.name2id(name, 'joint')
                adr = self.physics.model.jnt_qposadr[ind]
                if rot:
                    old_pos = self.physics.data.qpos[adr+3:adr+7]
                    self.physics.data.qpos[adr+3:adr+7] = pos
                else:
                    old_pos = self.physics.data.qpos[adr:adr+3]
                    self.physics.data.qpos[adr:adr+3] = pos
                self._type_cache[name] = 'joint'
            except Exception as e:
                item_type = 'body'

        if item_type == 'body':
            try:
                ind = self.physics.model.name2id(name, 'body')
                if rot:
                    old_pos = self.physics.data.xquat[ind]
                    self.physics.data.xquat[ind] = pos
                else:
                    old_pos = self.physics.data.xpos[ind]
                    self.physics.data.xpos[ind] = pos
                self.physics.model.body_pos[ind] = pos
                # old_pos = self.physics.model.body_pos[ind]
                item_type = 'body'
                self._type_cache[name] = 'body'
            except:
                item_type = 'unknown'
                print(('Could not shift item', name))

        if forward:
            self.physics.forward()


    def set_item_rot(self, name, rot, use_euler=False, mujoco_frame=True, forward=True):
        if use_euler or len(rot) == 3:
            rot = T.euler_to_quaternion(rot, 'wxyz')

        self.set_item_pos(name, rot, mujoco_frame, forward, True)


    def get_joints(self, jnts, sizes=None, vec=False):
        if vec:
            vals = []
        else:
            vals = {}

        for i, jnt in enumerate(jnts):
            if type(jnt) is not int:
                jnt = self.physics.model.name2id(jnt, 'joint')
            adr = self.physics.model.jnt_qposadr[jnt]
            size = 1
            if sizes is not None:
                size = sizes[i]

            if vec:
                vals.extend(self.physics.data.qpos[adr:adr+size])
            else:
                name = self.physics.model.id2name(jnt, 'joint')
                vals[name] = self.physics.data.qpos[adr:adr+size]
        return vals


    def set_joints(self, jnts, forward=True):
        for jnt, val in list(jnts.items()):
            if type(jnt) is not int:
                jnt = self.physics.model.name2id(jnt, 'joint')
            adr = self.physics.model.jnt_qposadr[jnt]
            offset = 1
            if hasattr(val, '__len__'):
                offset = len(val)
            self.physics.data.qpos[adr:adr+offset] = val
        if forward:
            self.physics.forward()


    def get_jnt_vec(self, jnts):
        if not len(jnts): return self.physics.data.qpos
        vals = []
        for name in jnts:
            ind = self.physics.model.name2id(name, 'joint')
            adr = self.physics.model.jnt_qposadr[ind]
            vals.append(adr)
        return self.physics.data.qpos[vals]


    def get_disp(self, body1, body2):
        pos1 = self.get_item_pos(body1)
        pos2 = self.get_itme_pos(body2)
        return pos2 - pos1


    def get_body_info(self):
        info = {}
        for i in range(self.physics.model.nbody):
            info[i] = {
                'name': self.physics.model.id2name(i, 'body'),
                'pos': self.physics.data.xpos[i],
                'quat': self.physics.data.xquat[i],
            }

        return info


    def get_jnt_info(self):
        info = {}
        dofadr = self.physics.model.jnt_dofadr
        for i in range(self.physics.model.njnt):
            inds = (dofadr[i], dofadr[i+1]) if i < self.physics.model.njnts-1 else (dofadr[i], self.physics.model.njnt)
            body_id = self.physics.model.jnt_bodyid[i]
            info[i] = {
                'name': self.physics.model.id2name(i, 'joint'),
                'angle': self.physics.data.qpos[inds[0]:inds[1]],
                'dofadr': inds,
                'body': self.physics.model.id2name(body_id, 'body'),
                'parent_body': self.physics.model.id2name(self.physics.model.body_parentid[body_id], 'body')
            }

        return info


    def get_geom_dimensions(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1):
        '''
        Geom type options:
        mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
        '''
        if geom_ind >= 0:
            return self.physics.model.geom_size[ind]

        inds = np.where(self.physics.model.geom_type == geom_type)
        return self.physics.model.geom_size[inds]


    def get_geom_positions(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1):
        '''
        Geom type options:
        mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
        '''
        if geom_ind >= 0:
            return self.physics.model.geom_pos[ind]
            
        inds = np.where(self.physics.model.geom_type == geom_type)
        return self.physics.data.geom_xpos[inds]


    # def get_geom_rotations(self, geom_type=enums.mjtGeom.mjGEOM_BOX, geom_ind=-1, use_euler=False):
    #     '''
    #     Geom type options:
    #     mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6, mjGEOM_MESH=7
    #     '''
    #     if geom_ind >= 0:
    #         return self.physics.model.geom_quat[ind]
            
    #     inds = np.where(self.physics.model.geom_type == geom_type)
    #     rots = self.physics.data.geom_xquat[inds]
    #     if use_euler:
    #         return np.array([T.quaternion_to_euler(r) for r in rots])
    #     return rots


    def get_camera_info(self, camera_name):
        ind = self.physics.model.name2id(camera_name, 'camera')
        fovy = self.physics.model.cam_fovy[ind].copy()
        pos = self.physics.data.cam_xpos[ind].copy()
        mat = self.physics.data.cam_xmat[ind].copy()
        return fovy, pos, mat


    def record_video(self, fname, actions=None, states=None, height=0, width=0, mode='position'):
        if not self.load_render:
            raise AssertionError('Cannot record video if the renderer is not loaded')
        elif actions is None and states is None:
            raise AssertionError('Must pass either action or state trajectory to record video')

        ims = []
        buf = actions if actions is not None else states
        for step in buf:
            if actions is not None: self.step(step, mode=mode)
            if states is not None: self.set_state(step)
            im = self.render(camera_id=camera_id, height=height, width=width, view=False)
            ims.append(im)
        np.save(fname, ims)


    def set_user_data(self, key, data):
        self._user_data[key] = data


    def get_user_data(self, key, default=None):
        return self._user_data.get(key, default)


    def compute_reward(self):
        return 0


    def is_done(self):
        return self._cur_iter >= self._max_iter


    def get_text_overlay(self, title='', body='', style='normal', position='top left'):
        return TextOverlay(title, body, style, position)


    def render(self, mode='rgb_array', height=0, width=0, camera_id=0,
               overlays=(), depth=False, scene_option=None, view=False,
               forward=False):
        if not self.load_render: return None
        # Make friendly with dm_control or gym interface
        depth = depth or mode == 'depth_array'
        view = view or mode == 'human'
        if height == 0: height = self.im_height
        if width == 0: width = self.im_wid
        if forward: self.physics.forward()

        pixels = None
        if self._cache_rendering:
            prev_x, prev_q = self._last_rendered_state
            x_changed = prev_x is None or np.any(np.abs(prev_x - self.physics.data.xpos) > 1e-5)
            q_changed = prev_q is None or np.any(np.abs(prev_q - self.physics.data.qpos) > 1e-5)
            if x_changed or q_changed:
                self._cached_images = {}
                self._last_rendered_state = (self.physics.data.xpos.copy(), self.physics.data.qpos.copy())
            elif (camera_id, height, width) in self._cached_images:
                pixels = self._cached_images[(camera_id, height, width)]

        if pixels is None:
            pixels = self.physics.render(height, width, camera_id, overlays, depth, scene_option)

            if self._cache_rendering: self._cached_images[(camera_id, height, width)] = pixels

        if view and self.use_viewer:
            self._render_viewer(pixels)

        return pixels


    def reset(self):
        self._cur_iter = 0
        self.physics.reset()
        # self._reload_viewer()
        self.ctrl_data = {}
        self.cur_time = 0.
        self.prev_time = 0.
        
        self.physics.data.qpos[:] = 0.
        self.physics.data.qvel[:] = 0.
        self.physics.data.qacc[:]= 0.
        self.physics.forward()
        return self.get_obs()


    def close(self):
        self.active = False
        if self._viewer is not None and self.use_glew:
            self._viewer.close()
            self._viewer = None
        self.physics.free()


    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)


    def list_joint_info(self):
        for i in range(self.physics.model.njnt):
            print('\n')
            print(('Jnt ', i, ':', self.physics.model.id2name(i, 'joint')))
            print(('Axis :', self.physics.model.jnt_axis[i]))
            print(('Dof adr :', self.physics.model.jnt_dofadr[i]))
            body_id = self.physics.model.jnt_bodyid[i]
            print(('Body :', self.physics.model.id2name(body_id, 'body')))
            print(('Parent body :', self.physics.model.id2name(self.physics.model.body_parentid[body_id], 'body')))
