import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
import random
from threading import Thread
import time

import robodesk

import opentamp.main as main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
import opentamp.core.util_classes.common_constants as const
from opentamp.core.util_classes.robots import Baxter
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from opentamp.pma.hl_solver import *
from opentamp.pma.pr_graph import *
# from opentamp.pma import backtrack_ll_solver_OSQP as bt_ll
# from opentamp.pma.robot_solver import RobotSolverOSQP
from opentamp.pma import backtrack_ll_solver_gurobi as bt_ll
from opentamp.pma.robot_solver import RobotSolverGurobi
import opentamp.core.util_classes.transform_utils as T

from opentamp.policy_hooks.utils.file_utils import load_config, setup_dirs, LOG_DIR
from opentamp.policy_hooks.run_training import argsparser
from opentamp.policy_hooks.utils.load_agent import *
import opentamp.policy_hooks.robodesk.hyp as hyp
import opentamp.policy_hooks.robodesk.desk_prob as prob


# def stepSimulation(t_limit=None, delay=0.):
#     import pybullet as op1
#     start_t = time.time()

#     p1 = bullet_client.BulletClient(op1.SHARED_MEMORY)
#     if p1 < 0:
#         print('Subprocess could not find PyBullet server, exiting...')
#         return 

#     time.sleep(0.1)
#     print('-----> Subprocess Connected to physics server')
#     while (t_limit is None or time.time() - start_t < t_limit):
#         p1.stepSimulation()
#         if delay > 0: time.sleep(delay)

if __name__ == '__main__':
    args = argsparser()
    args.config = 'opentamp.policy_hooks.robodesk.hyp'
    args.render = True

    base_config = hyp.refresh_config()
    base_config['id'] = 0
    base_config.update(vars(args))
    base_config['args'] = args
    config, config_module = load_config(args, base_config)
    config.update(base_config)
    config['load_render'] = True
    agent_config = load_agent(config)
    agent = build_agent(agent_config)
    env = agent.base_env
    mjc_env = agent.mjc_env

    '''
    agent.add_viewer()
    agent.mjc_env.set_attr('panda', 'right', np.zeros(7), forward=True)
    agent.render_viewer(env.render(resize=True))
    agent.render_viewer(env.render(resize=True))
    agent.render_viewer(env.render(resize=True))
    import ipdb; ipdb.set_trace()
    '''

    try:
        p.disconnect()
    except Exception as e:
        print(e)


    #const.NEAR_GRIP_COEFF = 5e-2
    #const.GRASP_DIST = 0.2
    #const.APPROACH_DIST = 0.025
    #const.EEREACHABLE_ROT_COEFF = 8e-3
    bt_ll.DEBUG = True
    openrave_bodies = None
    domain_fname = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/right_desk.domain"
    prob = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/probs/robodesk_prob.prob"
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FDSolver(d_c, cleanup_files=False)
    p_c = main.parse_file_to_dict(prob)
    visual = True # len(os.environ.get('DISPLAY', '')) > 0
    #visual = False
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
    params = problem.init_state.params


    cam_pos = (2.5, 111., -43., (0., 0., 0.)) # dist, yaw, pitch, target pos
    p.resetDebugVisualizerCamera(*cam_pos)
    # proc1 = Process(target=stepSimulation, args=(600, 0.))
    # proc1.name = 'PyBullet Sim Step'
    # proc1.start()

    sucs = []
    N_RUNS = 50
    for run_num in range(N_RUNS):
        agent.mjc_env.reset()

        for param in ['ball', 'upright_block', 'flat_block', \
                      'drawer_handle', 'shelf_handle', 'shelf', 'drawer']:
            pose = agent.mjc_env.get_item_pose(param, euler=True)
            targ = '{}_init_target'.format(param)
            params[param].pose[:,0] = pose[0]
            params[param].rotation[:,0] = pose[1]
            if targ in params:
                params[targ].value[:,0] = pose[0]
                params[targ].rotation[:,0] = pose[1]
            #params[param].pose[:,0] = env.physics.named.data.qpos[param][:3]
            #quat = env.physics.named.data.qpos[param][3:7]
            #quat = [quat[1], quat[2], quat[3], quat[0]]
            #euler = T.quaternion_to_euler(quat)
            #params[param].rotation[:,0] = euler
        #params['ball'].rotation[:,0] = [0., -0.4, 1.57]
        params['drawer'].hinge[:,0] = agent.mjc_env.get_attr('drawer', 'hinge')
        params['shelf'].hinge[:,0] = agent.mjc_env.get_attr('shelf', 'hinge')

        right_jnts = agent.mjc_env.get_attr('panda', 'right')
        lb, ub = params['panda'].geom.get_joint_limits('right')
        lb = np.array(lb) + 2e-3
        ub = np.array(ub) - 2e-3
        right_jnts = np.clip(right_jnts, lb, ub)
        params['panda'].right[:,0] = right_jnts 

        params['robot_init_pose'].right[:,0] = right_jnts
        params['robot_init_pose'].right_gripper[:,0] = agent.mjc_env.get_attr('panda', 'right_gripper')

        for param in params:
            targ = '{}_init_target'.format(param)
            if targ in params:
                params[targ].value[:,0] = params[param].pose[:,0]
                params[targ].rotation[:,0] = params[param].rotation[:,0]

        #goal = '(and (InSlideDoor ball drawer) (Stacked upright_block flat_block) (NearGripperRight panda green_button))'
        goal = '(and (SlideDoorClose shelf_handle shelf) (Stacked upright_block flat_block) (NearGripperRight panda green_button))'
        #goal = '(and (SlideDoorOpen drawer_handle drawer) (NearApproachRight panda upright_block))'
        #goal = '(and (InSlideDoor upright_block shelf) (NearApproachRight panda ball))'
        #goal = '(and (SlideDoorClose drawer_handle drawer) (InSlideDoor ball drawer))'
        #goal = '(and (InSlideDoor ball drawer) (InSlideDoor upright_block shelf) (SlideDoorClose drawer_handle drawer) (SlideDoorClose shelf_handle shelf))'
        #goal = '(Stacked upright_block flat_block)'
        #goal = '(and (SlideDoorClose shelf_handle shelf) (InSlideDoor upright_block shelf))'
        #goal = '(Lifted flat_block panda)'
        #goal = '(Lifted upright_block panda)'
        goal = '(Lifted ball panda)'
        #goal = '(SlideDoorClose shelf_handle shelf)'
        #goal = '(SlideDoorOpen drawer_handle drawer)'
        #goal = '(InSlideDoor ball drawer)'
        goal = '(At flat_block bin_target)'
        #goal = '(At ball bin_target)'
        #goal = '(InSlideDoor upright_block shelf)'
        #goal = '(InSlideDoor flat_block shelf)'
        #goal = '(and (At flat_block bin_target) (At upright_block off_desk_target))'
        #goal = '(and (InSlideDoor ball drawer) (NearGripperRight panda green_button))'
        #goal = '(NearGripperRight panda green_button)'
        #goal = '(Lifted upright_block panda)'
        #goal = "(NearGripperRight panda green_button)"

        goals = [
             # '(Lifted upright_block panda)',
             # '(Lifted ball panda)',
             '(Stacked upright_block flat_block)',
             #'(SlideDoorClose shelf_handle shelf)',
             #'(SlideDoorOpen drawer_handle drawer)',
             '(Near flat_block bin_target)',
             '(Near upright_block off_desk_target)',
             #'(InSlideDoor flat_block shelf)',
             #'(InGripperRight panda green_button)',
             # '(and (InSlideDoor upright_block shelf) (SlideDoorClose shelf_handle shelf))',
                ]

        goal = random.choice(goals)
        goal_info = []
        goal_str = goal.strip()[1:-1]
        if goal_str.startswith('and'):
            goals = goal_str.split('(')
            for g in goals:
                if g.find(')') >= 0:
                    goal_info.append(g.strip()[:-1].split(' '))
        else:
            goal_info = [goal_str.split(' ')]
        print('SOLVING:', goal, goal_info)

        print('CONSISTENT?', problem.init_state.is_consistent())
        solver = RobotSolverGurobi()

        plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5, max_iter=2)

        if type(plan) is str or plan is None:
            sucs.append([goal_info[0][0], 'OPT FAIL'])
            continue
        #import ipdb; ipdb.set_trace()

        if visual:
            agent.add_viewer()


        panda = plan.params['panda']
        for act in plan.actions:
            st, et = act.active_timesteps
            x = agent.get_state()
            for (pname, aname), inds in agent.state_inds.items():
                getattr(plan.params[pname], aname)[:, st] = x[inds]
            print('FAILED', act, plan.get_failed_preds((st,st)))
            for t in range(st, et):
                grip = panda.right_gripper[:, min(t+1, plan.horizon-1)]
                grip = -0.005 * np.ones(2) if grip[0] < 0.01 else 0.07 * np.ones(2)
                ctrl = np.r_[panda.right[:,t], grip]
                obs, rew, done, info = agent.mjc_env.step(ctrl)
                if 'hand_image' in obs:
                    agent.render_viewer(np.r_[obs['image'], obs['hand_image']])
                else:
                    agent.render_viewer(obs['image'])

        x = agent.get_state()
        goal_suc = [agent.parse_goal(x, g[0], g[1:]) for g in goal_info]
        print('Goal?', goal, goal_suc)
        #import ipdb; ipdb.set_trace()
        sucs.append([goal_info[0], goal_suc])

        print('\n\n-----------\nALL GOALS:')
        for item in sucs:
            print(item)
        print('------------\n\n')
        import ipdb; ipdb.set_trace()

