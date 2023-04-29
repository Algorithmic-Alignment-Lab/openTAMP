import copy
import pickle as pickle
import os
import sys
import time

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import array, float32
import seaborn as sns

FRAME = 10
TWINDOW = 300
TDELTA = 600 # 300
MIN_FRAME = 30
nan = np.nan
LOG_DIR = 'experiment_logs/'
prefix = os.path.expanduser('~')
SAVE_DIR = prefix+'/Dropbox/'
X_VARS = ['time', 'n_opt_calls', 'n_runs', 'n_learning_iters']
Y_VARS = ['n_success', 'opt_cost', 'tree_life']


def get_colors(n_colors):
    return cm.rainbow(np.linspace(0, 1, n_colors))


def get_test_data(keywords, include, exclude, pre=False, rerun=False, 
                  tdelta=TDELTA, wind=TWINDOW, lab='', lenthresh=0.99,
                  split_runs=False, label_vars=[]):
    exp_probs = os.listdir(LOG_DIR)
    all_data = {}
    for k in keywords:
        used = []
        all_data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                if dir_name.find(k) < 0 and dir_prefix.find(k) < 0:
                    continue
                d = dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue

                if len(include):
                    skip = True
                    for inc in include:
                        if dir_name.find(inc) >= 0 or dir_prefix.find(inc) >= 0:
                            skip = False
                    if skip: continue


                if len(exclude):
                    skip = False
                    for exc in exclude:
                        if dir_name.find(exc) >= 0 or dir_prefix.find(exc) >= 0:
                            skip = True
                            print(('skipping', dir_name))
                    if skip: continue

                full_dir = dir_prefix + dir_name
                full_exp = full_dir[:full_dir.rfind('_')]
                if full_exp in used: continue
                all_data[k][full_exp] = {}
                used.append(full_exp)
                i = 0
                data = []
                while i < 20:
                    cur_dir = '{0}_{1}'.format(full_exp, i)
                    if not os.path.isdir(cur_dir):
                        i += 1
                        continue
                    fnames = os.listdir(cur_dir)

                    if pre:
                        info = [f for f in fnames if f.find('hl_test_pre') >= 0 and f.endswith('pre_log.npy')]
                    elif rerun:
                        info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                    else:
                        info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                    if len(info):
                        for fname in info:
                            # print(('Loading data from', fname, full_dir))
                            try:
                                data.append(np.load(cur_dir+'/'+fname))
                            except Exception as e:
                                print('Skipping', fname, full_dir)
                                continue

                        label = gen_label(cur_dir, label_vars, split_runs, i)
                        all_data[k][full_exp][cur_dir] = {}
                        all_data[k][full_exp][cur_dir][cur_dir] = []
                        for buf in data:
                            for pts in buf:
                                pt = pts[0]
                                no, nt = int(pt[4]), int(pt[5])
                                #if label.find('cereal-and-milk') >= 0 and pt[10] > 1400: continue
                                #pt[1] = pt[1] * 20 # Path len
                                all_data[k][full_exp][cur_dir][cur_dir].append({'time': pt[3], 'success at end': pt[0], 'path length': 20. * pt[1], 'distance from goal': pt[2], 'n_data': pt[6], 'key': (no, nt), 'label': label, 'ind': i, 'success anywhere': pt[7], 'optimal_rollout_success': pt[9], 'number of plans': pt[10], 'subgoals anywhere': pt[11], 'subgoals closest distance': pt[12], 'collision': pt[8], 'exp id': i})
                                if len(pt) > 13:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['any target'] = pt[13]
                                if len(pt) > 14:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['smallest tolerance'] = pt[14]
                                if len(pt) > 16:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['success with postcond'] = pt[16]
                                if len(pt) > 17:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['success with adj_eta'] = pt[17]
                                if len(pt) > 18:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['episode return'] = pt[18]
                                if len(pt) > 19:
                                    all_data[k][full_exp][cur_dir][cur_dir][-1]['episode reward'] = pt[19]
                                # all_data[k][full_exp][cur_dir][cur_dir].append({'time': (pt[3]//tdelta+1)*tdelta, 'success at end': pt[0], 'path length': pt[1], 'distance from goal': pt[2], 'n_data': pt[6], 'key': (no, nt), 'description': label, 'ind': i, 'success anywhere': pt[7], 'optimal_rollout_success': pt[9], 'number of plans': pt[10]})

                    i += 1
    return all_data


def get_policy_data(policy, keywords=[], exclude=[], include=[]):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                full_dir = dir_prefix + dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                skip = False
                for ekey in exclude:
                    if full_dir.find(ekey) >= 0:
                        skip = True
        
                if len(include):
                    skip = True
                    for inc in include:
                        if dir_name.find(inc) >= 0 or dir_prefix.find(inc) >= 0:
                            skip = False
                    if skip: continue


                if len(exclude):
                    skip = False
                    for exc in exclude:
                        if dir_name.find(exc) >= 0 or dir_prefix.find(exc) >= 0:
                            skip = True
                            print(('skipping', dir_name))
                if skip: continue

                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                r = 'policy_{0}_log.txt'.format(policy)
                rollout_data = {}
                if not os.path.isfile(full_dir+'/'+r):
                    r = 'policy_{0}_log.pkl'.format(policy)
                if not os.path.isfile(full_dir+'/'+r): continue
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    next_data = str.split(next_data, '\n\n')
                    try:
                        r_data = [eval(d) for d in next_data if len(d)]
                    except:
                        continue
                    print('Loading {0} pts for {1}'.format(len(r_data), full_dir+'/'+r))
                    for pt in r_data:
                        pt['exp id'] = 0
                        if type(pt['train_loss']) is dict:
                            pt['train_loss'] = pt['train_loss']['loss']
                        if type(pt['val_loss']) is dict:
                            pt['val_loss'] = pt['val_loss']['loss']
                        if 'var' in pt and type(pt['var']) is dict:
                            pt['var'] = pt['var'][policy]
                    rollout_data[r] = r_data
                    data[k][full_exp][full_dir] = rollout_data

    return data


def get_motion_data(keywords=[], exclude=[], include=[]):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                full_dir = dir_prefix + dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                skip = False
                for ekey in exclude:
                    if full_dir.find(ekey) >= 0:
                        skip = True
        
                if len(include):
                    skip = True
                    for inc in include:
                        if dir_name.find(inc) >= 0 or dir_prefix.find(inc) >= 0:
                            skip = False
                    if skip: continue


                if len(exclude):
                    skip = False
                    for exc in exclude:
                        if dir_name.find(exc) >= 0 or dir_prefix.find(exc) >= 0:
                            skip = True
                            print(('skipping', dir_name))
                if skip: continue

                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                file_names = [fname for fname in file_names if fname.find('MotionInfo') >= 0]
                rollout_data = {'motion': []}
                for r in file_names:
                    with open(full_dir+'/'+r, 'r') as f:
                        next_data = f.read()
                    if len(next_data):
                        next_data = str.split(next_data, '\n\n')
                        try:
                            r_data = [eval(d) for d in next_data if len(d)]
                        except:
                            continue
                        for pt in r_data:
                            pt['exp id'] = 0
                            if full_exp.find('objs1') >= 0:
                                pt['description'] = 'cereal-only'
                            if full_exp.find('objs2') >= 0:
                                pt['description'] = 'cereal-and-milk'
                            if full_exp.find('objs4') >= 0:
                                pt['description'] = 'all'
                            
                        print('MOTION: Loading {0} pts for {1}'.format(len(r_data), full_dir+'/'+r))
                        rollout_data['motion'].extend(r_data)
                        data[k][full_exp][full_dir] = rollout_data

    return data


def get_rollout_info_data(keywords=[], exclude=[], include=[]):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                full_dir = dir_prefix + dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                skip = False
                for ekey in exclude:
                    if full_dir.find(ekey) >= 0:
                        skip = True
        
                if len(include):
                    skip = True
                    for inc in include:
                        if dir_name.find(inc) >= 0 or dir_prefix.find(inc) >= 0:
                            skip = False
                    if skip: continue


                if len(exclude):
                    skip = False
                    for exc in exclude:
                        if dir_name.find(exc) >= 0 or dir_prefix.find(exc) >= 0:
                            skip = True
                            print(('skipping', dir_name))
                if skip: continue

                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                file_names = [fname for fname in file_names if fname.find('RolloutInfo') >= 0]
                rollout_data = {'rollout': []}
                for r in file_names:
                    with open(full_dir+'/'+r, 'r') as f:
                        next_data = f.read()
                    if len(next_data):
                        next_data = str.split(next_data, '\n\n')
                        try:
                            r_data = [eval(d) for d in next_data if len(d)]
                        except:
                            continue
                        for pt in r_data:
                            pt['exp id'] = 0
                            goal_vals = pt.get('per_goal_success', {'basegoal': 0.})
                            for goal in goal_vals:
                                new_pt = copy.copy(pt)
                                new_pt['goal'] = goal
                                new_pt['success rate'] = goal_vals[goal]
                                rollout_data['rollout'].append(new_pt)
                        print('ROLLOUT: Loading {0} pts for {1}'.format(len(r_data), full_dir+'/'+r))
                        #rollout_data['rollout'].extend(r_data)
                        data[k][full_exp][full_dir] = rollout_data

    return data


def get_rollout_data(keywords=[], nfiles=20, exclude=[]):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                full_dir = dir_prefix + dir_name
                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                rollout_logs = ['rollout_log_{0}_True.txt'.format(i) for i in range(nfiles)]# [f for f in file_names if f.startswith('rollout')]
                rollout_logs += ['rollout_log_{0}_False.txt'.format(i) for i in range(nfiles)]
                rollout_data = {}
                for r in rollout_logs:
                    if not os.path.isfile(full_dir+'/'+r): continue
                    with open(full_dir+'/'+r, 'r') as f:
                        next_data = f.read()
                    if len(next_data):
                        try:
                            r_data = eval(next_data)
                        except:
                            continue
                        for pt in next_data:
                            pt['exp id'] = 0
                        rollout_data[r] = r_data
                    else:
                        print(('no data for', r))
                data[k][full_exp][full_dir] = rollout_data

    return data


def gen_first_success_plots(x_var='time'):
    exp_probs = os.listdir(LOG_DIR)
    master_plot = []
    all_data = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('rollout') and f.endswith('False.txt')]
            rollout_data = {}
            ts_to_data = {}
            for i, r in enumerate(rollout_logs):
                print((exp_name, dir_name, r))
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    costs = eval(next_data)
                    if 'avg_first_success' not in costs[0]: continue
                    for j, c in enumerate(costs):
                        c['ind'] = np.mean(c['avg_first_success'])
                        if j not in ts_to_data:
                            ts_to_data[j] = []
                        ts_to_data[j].append(c)
                    rollout_data[r] = costs
            xs = [np.mean([c[x_var] for c in ts_to_data[n]]) for n in ts_to_data]
            ys = [np.mean([c['ind'] for c in ts_to_data[n]]) for n in ts_to_data]
            all_data.append((exp_name+dir_name, xs, ys))

    plt.title('Steps to first success')
    plt.xlabel(x_var)
    plt.ylabel('Avg. step of first success')
    colors = get_colors(len(all_data))
    for i, (label, xs, ys) in enumerate(all_data):
        plt.plot(xs, ys, label=label, color=colors[i])
    plt.savefig(SAVE_DIR+'/goal_vs_{0}.png'.format(x_var), pad_inches=0.01)
    plt.clf()


def get_td_loss(keywords=[], exclude=[], pre=False):
    tdelta = 5
    exp_probs = os.listdir(LOG_DIR)
    exp_data = {}
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        # exp_data = []
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = True
                for k in keywords:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue

            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:-1]
            i = 0
            data = []
            while i < 20:
                if not os.path.isdir(full_exp+str(i)):
                    i += 1
                    continue
                fnames = os.listdir(full_exp+str(i))
                info = [f for f in fnames if f.find('td_error') >= 0 and f.endswith('npy') and f.find('test') < 0]
                if len(info):
                    cur_data = []
                    for step in info:
                        cur_data.append(np.load(full_exp+str(i)+'/'+step))
                    dlen = max([len(dt) for dt in cur_data])
                    data.append([])
                    for n in range(dlen):
                        data[-1].append(np.mean([cur_data[ind][n] for ind in range(len(cur_data)) if n < len(cur_data[ind])], axis=0))
                    data[-1] = np.array(data[-1])
                i += 1

            if not len(data):
                print(('skipping', full_exp))
                continue
            dlen = min([len(d) for d in data])
            dmax = max([len(d) for d in data])
            print(('Gathering data for', full_exp, 'length:', dlen, 'all len:', [len(d) for d in data]))
            end = False
            cur_t = 0
            while not end:
                end = True
                for d in data:
                    next_frame = d[cur_t:cur_t+tdelta] # [pt[0] for pt in d if pt[0,3] >= cur_t and pt[0,3] <= cur_t + TWINDOW]
                    if len(next_frame):
                        end = False
                        if len(next_frame) >= MIN_FRAME:
                            next_pt = np.mean(next_frame, axis=0)
                            no, nt = 0, 0 # int(next_pt[4]), int(next_pt[2])
                            if (no, nt) not in exp_data:
                                exp_data[no, nt] = []
                            exp_data[no, nt].append((full_exp, cur_t, next_pt[0]))
                cur_t += tdelta


            '''
            for i in range(dmax - FRAME):
                cur_t = np.mean([d[i:i+FRAME,:,3] for d in data if i+FRAME < len(d)])

                for d in data:
                    if len(d) < i+FRAME: continue
                    cur_fr = np.mean(d[i:i+FRAME], axis=0)
                    for pt in cur_fr:
                        val = pt[0]
                        # cur_t = pt[3]
                        nt = int(pt[2])
                        no = int(pt[4])
                        if (no, nt) not in exp_data:
                            exp_data[no, nt] = []
                        exp_data[no, nt].append((full_exp, cur_t, val))
            '''
        for no, nt in exp_data:
            print('Plotting', no, nt, exp_name)
            pd_frame = pd.DataFrame(exp_data[no, nt], columns=['exp_name', 'time', 'value'])
            sns.set()
            sns_plot = sns.relplot(x='time', y='value', hue='exp_name', kind='line', data=pd_frame)
            keyid = ''
            for key in keywords:
                keyid += '_{0}'.format(key)
            pre_lab = '_pre' if pre else ''
            sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_td_error{2}{3}.png'.format(no, nt, keyid, pre_lab))


def get_fail_info(keywords=[], exclude=[], pre=False, rerun=False, xvar='time', avg_time=True, tdelta=TDELTA, wind=TWINDOW, lab='', lenthresh=0.99, label_vars=[], include=[], max_t=14400):
    exp_probs = os.listdir(LOG_DIR)
    exp_data = {}
    exp_len_data = {}
    exp_dist_data = {}
    exp_true_data = {}
    targets = {}
    used = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = False
                for k in keywords:
                    if dir_name.find(k) < 0 and dir_prefix.find(k) < 0:
                        skip = True
                if skip: continue

            if len(include):
                skip = True
                for k in include:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue


            print(dir_name)
            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                        print(('skipping', dir_name))
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:full_dir.rfind('_')]
            if full_exp in used: continue
            used.append(full_exp)
            i = 0
            data = []
            while i < 20:
                cur_dir = '{0}_{1}'.format(full_exp, i)
                if not os.path.isdir(cur_dir):
                    i += 1
                    continue
                fnames = os.listdir(cur_dir)

                info = [f for f in fnames if f.find('failure') >= 0 and f.endswith('data.txt')]
                if len(info):
                    for fname in info:
                        print(('Loading data from', fname))
                        with open(cur_dir+'/'+fname, 'r') as f:
                            data.append(f.read().splitlines())

                    label = gen_label(cur_dir, label_vars)
                    for buf in data:
                        for pts in buf:
                            pts = eval(pts)
                            no, nt = int(pts['no']), int(pts['nt'])
                            if (no,nt) not in exp_data: exp_data[no,nt] = []
                            if (no,nt) not in targets: targets[no,nt] = []
                            pts['exp_name'] = label
                            exp_data[no,nt].append(pts)
                            targs = pts['goal']
                            targinfo = []
                            for targind in range(len(targs)):
                                if targs[targind] == 1:
                                    targinfo.append(targind)
                            targets[no,nt].append([tuple(targinfo)])
                i += 1

    for no, nt in targets:
        print(('Plotting', no, nt, exp_name))
        pd_frame = pd.DataFrame(targets[no, nt], columns=['targets']) #'target_{0}'.format(i) for i in range(no)])
        print((pd_frame['targets'].value_counts()[:10]))
        sns.set()
        plt.clf()
        sns_plot = sns.countplot(x="value", hue="variable", data=pd.melt(pd_frame))
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        plt.savefig(SAVE_DIR+'/{0}obj_{1}targ_failedtargets{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))
        plt.clf()


def get_hl_tests(keywords=[], exclude=[], pre=False, rerun=False, xvar='time', avg_time=True, tdelta=TDELTA, wind=TWINDOW, lab='', lenthresh=0.99, label_vars=[], include=[], max_t=14400):
    exp_probs = os.listdir(LOG_DIR)
    all_data = []
    exp_data = {}
    exp_len_data = {}
    exp_dist_data = {}
    exp_true_data = {}
    used = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = False
                for k in keywords:
                    if dir_name.find(k) < 0 and dir_prefix.find(k) < 0:
                        skip = True
                if skip: continue

            if len(include):
                skip = True
                for k in include:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue


            print(dir_name)
            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                        print(('skipping', dir_name))
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:full_dir.rfind('_')]
            if full_exp in used: continue
            used.append(full_exp)
            i = 0
            data = []
            while i < 20:
                cur_dir = '{0}_{1}'.format(full_exp, i)
                if not os.path.isdir(cur_dir):
                    i += 1
                    continue
                fnames = os.listdir(cur_dir)

                if pre:
                    info = [f for f in fnames if f.find('hl_test_pre') >= 0 and f.endswith('pre_log.npy')]
                elif rerun:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                else:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                if len(info):
                    for fname in info:
                        print(('Loading data from', fname))
                        data.append(np.load(cur_dir+'/'+fname))

                    label = gen_label(cur_dir, label_vars)
                    label = label.replace('_', ' ')
                    for buf in data:
                        for pts in buf:
                            pt = pts[0]
                            no, nt = int(pt[4]), int(pt[5])
                            if (no,nt) not in exp_data: exp_data[no,nt] = []
                            if (no,nt) not in exp_len_data: exp_len_data[no,nt] = []
                            if (no,nt) not in exp_dist_data: exp_dist_data[no,nt] = []
                            if (no,nt) not in exp_true_data: exp_true_data[no,nt] = []

                            if xvar == 'time':
                                xval = (pt[3] // tdelta) * tdelta
                                if xval < max_t:
                                    exp_data[no, nt].append((label, xval, pt[0]))
                                    exp_data[no, nt].append((label, xval+tdelta, pt[0]))
                                    if pt[0] > lenthresh:
                                        exp_len_data[no, nt].append((label, xval, pt[1]))
                                    if pt[0] <= 0.5: exp_dist_data[no, nt].append((label, xval, pt[2]))
                                    if len(pt) > 7: exp_true_data[no, nt].append((label, xval, pt[7]))
                            elif rerun:
                                exp_data[no, nt].append((label, pt[-1], pt[0]))
                                if pt[0] > lenthresh:
                                    exp_len_data[no, nt].append((label, pt[-1], pt[1]))
                            elif xvar == 'N' or xvar == 'n_data':
                                exp_data[no, nt].append((label, pt[6], pt[0]))
                                if pt[0] > lenthresh:
                                    exp_len_data[no, nt].append((label, pt[6], pt[1]))
                            all_data.append({'time': (pt[3]//tdelta)*tdelta, 'value': pt[0], 'len': pt[1], 'dist': pt[2], 'N': pt[6], 'problem': '{0} object {1} goals'.format(no, nt), 'description': label, 'ind': i})

                i += 1

    pd_frame = pd.DataFrame(all_data, columns=['time', 'description', 'N', 'value', 'len', 'dist', 'problem', 'ind'])
    # pd_frame = pd_frame.groupby(['time', 'description', 'problem', 'ind'], as_index=False).mean()
    sns.set()
    sns.set_context('paper', font_scale=1.5)
    plt.title('3 object with 3 goals')
    fig = plt.figure(figsize=(10,6))
    axs = fig.subplots(ncols=3)
    sns_plot = sns.relplot(x=xvar, y='value', hue='description', row='problem', kind='line', data=pd_frame)
    sns_plot.fig.set_figwidth(10)
    sns_plot._legend.remove()
    sns_plot.fig.get_axes()[0].legend(loc=(0., -0.6), prop={'size': 15})
    sns_plot.fig.axes[0].set_title('value')

    l, b, w, h = sns_plot.fig.axes[0]._position.bounds
    sns_plot.fig.add_axes((l+w+0.1, b, w, h))
    sns_plot_2 = sns.relplot(x=xvar, y='dist', hue='description', row='problem', kind='line', data=pd_frame, legend=False, ax=sns_plot.fig.axes[1])
    sns_plot.fig.axes[1].set_title('distance')

    l, b, w, h = sns_plot.fig.axes[1]._position.bounds
    sns_plot.fig.add_axes((l+w+0.1, b, w, h))
    sns_plot_2 = sns.relplot(x=xvar, y='len', hue='description', row='problem', kind='line', data=pd_frame, legend=False, ax=sns_plot.fig.axes[2])
    sns_plot.fig.axes[2].set_title('length')
    keyid = ''
    for key in keywords:
        keyid += '_'+str(key)
    sns_plot.fig.savefig(SAVE_DIR+'/allgraphs_{0}_{1}.png'.format(keyid, lab), bbox_inches="tight")

    for no, nt in exp_data:
        print(('Plotting', no, nt, exp_name))
        pd_frame = pd.DataFrame(exp_data[no, nt], columns=['exp_name', xvar, 'value'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='value', hue='exp_name', kind='line', data=pd_frame)
        sns_plot._legend.remove()
        sns_plot.fig.get_axes()[0].legend(loc=(1.5, 1.5))
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.fig.savefig(SAVE_DIR+'/{0}obj_{1}targ_val{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab), bbox_inches="tight")

    for no, nt in exp_len_data:
        print(('Plotting', no, nt, exp_name))
        pd_frame = pd.DataFrame(exp_len_data[no, nt], columns=['exp_name', xvar, 'length'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='length', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_len{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))

    for no, nt in exp_dist_data:
        print(('Plotting', no, nt, exp_name))
        pd_frame = pd.DataFrame(exp_dist_data[no, nt], columns=['exp_name', xvar, 'disp'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='disp', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_disp{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))


    for no, nt in exp_true_data:
        print(('Plotting', no, nt, exp_name))
        pd_frame = pd.DataFrame(exp_true_data[no, nt], columns=['exp_name', xvar, 'true'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='true', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:-1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_true{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))


def plot(data, columns, descr, xvars, yvars, separate=True, keyind=0, inter=100, rolling=True, window=100, xlim=None, ylim=None, fname='', witherr=False, style=None):
    sns.set()
    for k in data:
        if not len(data[k]): continue
        pd_frame = pd.DataFrame(data[k], columns=columns)
        datetime_var = xvars[0]+'_datetime'
        pd_frame[datetime_var] = pd.to_datetime(pd_frame[xvars[0]], unit='s')
        if 'time' in pd_frame: pd_frame['time'] = pd_frame['time'] / 3600.
        if 'path length' in pd_frame:
            pd_frame = pd_frame[pd_frame['success at end'] == 1.]
        pd_frame.set_index(datetime_var, inplace=True)
        pd_frame.sort_index(inplace=True)
        #pd_frame.reset_index(inplace=True)
        if 'goal' in columns:
            base_frame = pd_frame
            goals = pd_frame['goal'].unique()
            group_frame = pd_frame.groupby('goal')
            for goal_ind, goal in enumerate(goals):
                pd_frame = group_frame.get_group(goal)
                nrows = len(pd_frame.index)
                numeric_cols = [col for col in pd_frame.columns if pd_frame[col].dtype.name != 'object' and col not in xvars and col != datetime_var]
                print('Rolling mean on', numeric_cols)
                dfs = [pd.DataFrame(y) for x, y in pd_frame.groupby(['exp id', 'description'], as_index=False)]
                for df in dfs:
                    wind = int(window)
                    rolling = df[numeric_cols].rolling(wind, win_type=None, min_periods=1).mean()
                    for col in numeric_cols: df[col] = rolling[col]
                pd_frame = pd.concat(dfs)
                leg_labels = getattr(pd_frame, columns[0]).unique()
                sns_plot = None
                for xv in xvars:
                    for yind, yv in enumerate(yvars):
                        print('Plotting', xv, yv)
                        cur_y = yv
                        inter = int(inter)
                        dashes = [(1,0), (4,2), (1,4), (2,2)]
                        df = pd_frame
                        ci = 'sd' # if witherr else None
                        err_style='band'
                        err_kws = {'alpha': 0.12}
                        scatter_kws={"s": 50}
                        if sns_plot is None:
                            sns_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=style, kind='line', data=df, markers=False, dashes=dashes, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws, scatter_kws=scatter_kws)
                            sns_plot.fig.set_figwidth(10)
                            sns_plot._legend.remove()
                            # sns_plot.fig.get_axes()[0].legend(loc=(0.0, -0.5), prop={'size': 12})
                            cur_plot = sns_plot
                        else:
                            l, b, w, h = sns_plot.fig.axes[-1]._position.bounds
                            sns_plot.fig.add_axes((l+w+0.1, b, w, h))
                            sub_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=style, kind='line', data=df, legend=False, ax=sns_plot.fig.axes[-1], dashes=dashes, markers=False, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws)
                        #sns_plot.fig.axes[-1].set_title('Robosuite', size=14)
                        sns_plot.fig.axes[-1].set(xlabel='Training Time (seconds)')
                        sns_plot.fig.axes[-1].set(ylabel='Success Rate')
                        if xlim is not None:
                            sns_plot.fig.axes[-1].set(xlim=xlim[xind])
                            sns_plot.fig.axes[-1].setx_ticks([0,1,2,3,4])
                        if ylim is not None:
                            sns_plot.fig.axes[-1].set(ylim=ylim[yind])
                            sns_plot.fig.axes[-1].set_yticks(np.arange(0, ylim[yind][1], ylim[yind][1]/10.))
                sns_plot.fig.axes[-1].set_xticks([0,1,2,3,4])
                sns_plot.fig.get_axes()[0].tick_params(axis='both', which='major', labelsize=20)
                naxs = len(sns_plot.fig.get_axes())
                sns_plot.fig.get_axes()[0].legend(bbox_to_anchor=[-0.05, -0.15], loc='upper left', prop={'size': 12})
                sns_plot.fig.get_axes()[0].xlabel('Training Time (seconds)')
                sns_plot.fig.get_axes()[0].ylabel('Success Rate')
                #sns_plot.fig.get_axes()[0].title('Robosuite')
                sns_plot.savefig(SAVE_DIR+'/{0}{1}_{2}_goal_{3}.png'.format(fname, k, descr.replace(' ', '_'), goal_ind))
                print(('PLOTTED for', k, descr))
                sns.set()
        else:
            if rolling:
                nrows = len(pd_frame.index)
                numeric_cols = [col for col in pd_frame.columns if pd_frame[col].dtype.name != 'object' and col not in xvars and col != datetime_var]
                print('Rolling mean on', numeric_cols)
                #dfs = [pd.DataFrame(y) for x, y in pd_frame.groupby(['exp id', 'description'], as_index=False)]
                dfs = [pd.DataFrame(y) for x, y in pd_frame.groupby(['exp id', '# Objects'], as_index=False)]
                for df in dfs:
                    wind = int(window)
                    #wind = '{}s'.format(int(window))
                    #wind = '1200s'
                    rolling = df[numeric_cols].rolling(wind, win_type=None, min_periods=1).mean()
                    for col in numeric_cols: df[col] = rolling[col]
                pd_frame = pd.concat(dfs)
            leg_labels = getattr(pd_frame, columns[0]).unique()
            sns_plot = None
            for xind, xv in enumerate(xvars):
                for yind, yv in enumerate(yvars):
                    print('Plotting', xv, yv)
                    cur_y = yv
                    inter = int(inter)
                    dashes = [(1,0), (4,2), (1,4), (2,2)]
                    df = pd_frame
                    if type(yv) not in (np.string_, str):
                        df = pd_frame.melt(id_vars=[xv, columns[0]], value_vars=yv, var_name='y_variable', value_name='value')
                        style = 'y_variable'
                        cur_y = 'value'
                        print(('Combining', yv))

                    ci = None # 'sd' # if witherr else None
                    err_style='band'
                    err_kws = {'alpha': 0.12}
                    markers = {
                            'can-only': 'o', 'cereal-and-milk': 's', 'all': '^', 'all-noisy': 'X'}
                    if sns_plot is None:
                        print(df.columns)
                        #sns_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=columns[0], kind='line', data=df, markers=True, dashes=dashes, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws, palette=sns.color_palette(['black', 'dimgrey', 'silver', 'white'], n_colors=2))
                        #sns_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=columns[0], kind='line', data=df, dashes=False, markers=markers, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws, palette=sns.color_palette(['silver', 'grey', 'dimgrey', 'black'], n_colors=4), hue_order=['cereal-only', 'cereal-and-milk', 'all-objects', 'all-noisy'])
                        sns_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=columns[0], kind='line', data=df, dashes=False, markers=markers, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws, palette=sns.color_palette('colorblind', n_colors=4), hue_order=['can-only', 'cereal-and-milk', 'all', 'all-noisy'], linewidth=3, markersize=10)
                        sns_plot.fig.set_figwidth(10)
                        sns_plot._legend.remove()
                        # sns_plot.fig.get_axes()[0].legend(loc=(0.0, -0.5), prop={'size': 12})
                        cur_plot = sns_plot
                    else:
                        l, b, w, h = sns_plot.fig.axes[-1]._position.bounds
                        sns_plot.fig.add_axes((l+w+0.1, b, w, h))
                        sub_plot = sns.relplot(x=xv, y=cur_y, hue=columns[0], style=style, kind='line', data=df, legend=False, ax=sns_plot.fig.axes[-1], dashes=dashes, markers=False, ci=ci, n_boot=100, err_style=err_style, err_kws=err_kws)
                    #sns_plot.fig.axes[-1].set_title('{0} vs {1}'.format(xv, cur_y), size=14)
                    if xlim is not None:
                        sns_plot.fig.axes[-1].set(xlim=xlim[xind])
                    if ylim is not None:
                        sns_plot.fig.axes[-1].set(ylim=ylim[yind])
                        #sns_plot.fig.axes[-1].set_yticks(np.arange(0, ylim[yind][1], ylim[yind][1]/10.))
                        sns_plot.fig.axes[-1].set_yticks(np.arange(0, ylim[yind][1], ylim[yind][1]/5.))
            if 'time' in pd_frame:
                sns_plot.fig.axes[0].set_xticks([0,1,2,3,4])
            else:
                sns_plot.fig.axes[0].set_xticks([0,1000,2000])
            sns_plot.fig.get_axes()[0].tick_params(axis='both', which='major', labelsize=18)
            sns_plot.fig.get_axes()[0].set(xlabel='')
            sns_plot.fig.get_axes()[0].set(ylabel='')
            #sns_plot.fig.get_axes()[0].set_title('Robosuite')
            naxs = len(sns_plot.fig.get_axes())
            sns_plot.fig.get_axes()[0].legend(bbox_to_anchor=[1., -0.], loc='lower right', prop={'size': 18, 'family': 'serif'}, facecolor='white', title="")
            handles, labels = sns_plot.fig.get_axes()[0].get_legend_handles_labels()       
            sns_plot.fig.get_axes()[0].legend(handles=handles[1:], labels=labels[1:], facecolor="white", framealpha=0.85, prop={'size': 16, 'family': 'serif'})
            sns_plot.savefig(SAVE_DIR+'/{0}{1}_{2}.png'.format(fname, k, descr.replace(' ', '_')))
            print(('PLOTTED for', k, descr))
            sns.set()

    if 'path length' in df:
        print('LENGTHS:')
        vals = df['# Objects'].unique()
        for val in vals:
            print(val, df[df['# Objects'] == val]['path length'].mean())
        import ipdb; ipdb.set_trace()
    if 'episode return' in df:
        print('RETS:')
        vals = df['# Objects'].unique()
        for val in vals:
            print(val, df[df['# Objects'] == val]['episode return'][-50:].mean())
            print(val, df[df['# Objects'] == val]['episode reward'][-50:].mean())
        import ipdb; ipdb.set_trace()


def gen_label(exp_dir, label_vars=[], split_runs=False, run_ind=0):
    if not os.path.isfile(exp_dir+'/args.pkl'):
        return exp_dir[:exp_dir.rfind('_')]
    label = ''
    with open(exp_dir+'/args.pkl', 'rb') as f:
        args = pickle.load(f)
    args = vars(args)
    if not len(label_vars):
        label = args['descr'].replace('_', ' ')

    for v in label_vars:
        if v not in args: continue
        if args[v].find('nois') >= 0:
            label = 'all-noisy'
            break
        elif args[v].find('2obj') >= 0:
            label = 'cereal-and-milk'
            break
        elif args[v].find('4obj') >= 0:
            label = 'all'
            break
        elif args[v].find('hier') >= 0:
            label = 'cereal-only'
            break
        elif args[v].find('can') >= 0:
            label = 'can-only'
            break

        if v == 'descr':
            label += '{0}'.format(str(args[v]).replace('_', ' '))
        else:
            label += ' {0}: {1}'.format(v, str(args[v]).replace('_', ' '))

    if split_runs: label += ' - run {0}'.format(run_ind)
    return label


def gen_data_plots(xvar, yvar, keywords=[], lab='rollout', inter=1., 
                   label_vars=[], ylabel='value', separate=True, keyind=3, 
                   exclude=[], include=[], split_runs=False,
                   pre=False, rolling=True, window=100, xlim=None, ylim=None,
                   fname='', split_goal=False):
    if not len(keywords): keywords.append(LOG_DIR)
    if lab == 'rollout':
        rd = get_rollout_data(keywords, exclude=exclude)
    elif lab == 'motion':
        rd = get_motion_data(keywords, exclude=exclude, include=include)
    elif lab == 'rollout_info':
        rd = get_rollout_info_data(keywords, exclude=exclude, include=include)
    elif lab == 'test':
        rd = get_test_data(keywords, include=include, exclude=exclude, split_runs=split_runs, pre=pre, label_vars=label_vars)
    else:
        rd = get_policy_data(lab, keywords, exclude=exclude,include=include)
    prefix = lab
    data = {}
    print('Collected data...')
    xvars = [xvar] if type(xvar) is not list else xvar
    yvars = [yvar] if type(yvar) is not list else yvar
    flat_yvars = np.concatenate([yv for yv in yvars if type(yv) is list]+[[yv] for yv in yvars if type(yv) is not list]).tolist()
    inds_to_var = {}
    for keyword in rd:
        key_data = []
        new_data = rd[keyword]
        for fullexp in rd[keyword]:
            e = new_data[fullexp]
            for ename in e:
                base_label = gen_label(ename, label_vars)
                curexp = e[ename]
                inds_to_var = {}
                for rname in curexp:
                    r = curexp[rname]
                    if not len(r): continue
                    if any(xvar not in r[0] for xvar in xvars) or any([v not in r[0] for v in flat_yvars]): continue
                    for pt in r:
                        label = pt.get('label', base_label)
                        xvals = []
                        for xvar in xvars:
                            xvals.append(pt[xvar]//inter*inter)
                            #xvals.append(pt[xvar])
                        yvals = []
                        yinds = []
                        for v in yvars:
                            if type(v) is not str:
                                if hasattr(pt[v[0]], '__len__'):
                                    for i in range(len(pt[v[0]])):
                                        for j in range(len(v)):
                                            yvals.append(pt[v[j]][i])
                                    for j in range(len(v)):
                                        inds_to_var[v[j]] = inds_to_var.get(v[j], len(pt[v[j]]))
                                else:
                                    for j in range(len(v)):
                                        yvals.append(pt[v[j]])
                                        inds_to_var[v[j]] = 1

                            else:
                                if hasattr(pt[v], '__len__'):
                                    for i in range(len(pt[v])):
                                        yvals.append(pt[v][i])
                                    inds_to_var[v] = inds_to_var.get(v, len(pt[v]))
                                else:
                                    yvals.append(pt[v])
                                    inds_to_var[v] = 1
                        if split_goal:
                            key_data.append([label, keyword, pt['exp id'], pt['goal']]+xvals+yvals)
                        else:
                            key_data.append([label, keyword, pt['exp id']]+xvals+yvals)
            print(('Set data for', keyword, fullexp))
        data[keyword] = key_data
    yvar_labs = []
    flat_yvar_labs = []
    for v in yvars:
        if type(v) is not str:
            for i in range(inds_to_var.get(v[0],1)):
                yvar_labs.append([])
                for subv in v:
                    yvar_labs[-1].append(subv+'{0}'.format('_'+str(i) if inds_to_var.get(subv,0) > 1 else ''))
                    flat_yvar_labs.append(subv+'{0}'.format('_'+str(i) if inds_to_var.get(subv,0) > 1 else ''))
        else:
            yvar_labs.extend([v+'{0}'.format('_'+str(i) if inds_to_var.get(v,0) > 1 else '') for i in range(inds_to_var.get(v,1))])
            flat_yvar_labs.extend([v+'{0}'.format('_'+str(i) if inds_to_var.get(v,0) > 1 else '') for i in range(inds_to_var.get(v,1))])

    # yvar_labs = np.concatenate([[v+'{0}'.format('_'+str(i) if inds_to_var.get(v, 0) > 1 else '') for i in range(inds_to_var.get(v, 1))] for v in yvars])
    if split_goal:
        plot(data, ['description', 'key', 'exp id', 'goal']+xvars+flat_yvar_labs, '{0}_{1}_vs_{1}'.format(prefix, xvar, ylabel), xvars, yvar_labs, separate=separate, keyind=keyind, inter=inter, rolling=rolling, window=window, xlim=xlim, ylim=ylim, fname=fname)
    else:
        #plot(data, ['description', 'key', 'exp id']+xvars+flat_yvar_labs, '{0}_{1}_vs_{1}'.format(prefix, xvar, ylabel), xvars, yvar_labs, separate=separate, keyind=keyind, inter=inter, rolling=rolling, window=window, xlim=xlim, ylim=ylim, fname=fname)
        print(data.keys())
        plot(data, ['# Objects', 'key', 'exp id']+xvars+flat_yvar_labs, '{0}_{1}_vs_{1}'.format(prefix, xvar, ylabel), xvars, yvar_labs, separate=separate, keyind=keyind, inter=inter, rolling=rolling, window=window, xlim=xlim, ylim=ylim, fname=fname)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Loading command line:', sys.argv)
        perpetual = eval(sys.argv[1])
    else: 
        perpetual = False

    if len(sys.argv) > 2:
        keywords = [sys.argv[2]]
    else: 
        keywords=[]

    if len(sys.argv) > 3:
        include = sys.argv[3:]
    else: 
        include=[]#['conttask_grip']

    exclude=[]
    terminate = False
    while not terminate:
        if not perpetual:
            terminate = True
        gen_data_plots(xvar='time', yvar=['success at end'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=450, window=900, ylim=[(0.,1.), (0.,1.), (0, 1.), (0, 2.)], xlim=[(0, 14000)], fname='endsucc_{}'.format(keywords[0]))
        gen_data_plots(xvar='time', yvar=['episode return', 'episode reward'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=200, window=100, fname='endreturn_{}'.format(keywords[0]))
        gen_data_plots(xvar='time', yvar=['success at end', 'path length'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=600, window=600, xlim=[(0., 4.)], fname='pathlen_{}'.format(keywords[0]))
        gen_data_plots(xvar='time', yvar=['success at end'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=600, window=600, xlim=[(0., 4.)], ylim=[(0.,1.), (0.,1.), (0, 1.), (0, 2.)], fname='endsucc_{}'.format(keywords[0]))
        gen_data_plots(xvar='number of plans', yvar=['success at end'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=100, window=600, ylim=[(0.,1.), (0.,1.), (0, 1.), (0, 2.)], fname='endsucc_nplans_{}'.format(keywords[0]), xlim=[(0,2000)])
        #gen_data_plots(xvar='time', yvar=['success at end'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=True, include=include, inter=120, window=200, ylim=[(0.,1.), (0.,1.), (0, 1.), (0, 2.)], fname='splitendsucc_{}'.format(keywords[0]))

        gen_data_plots(xvar='time', yvar=['opt duration per ts'], keywords=keywords, lab='motion', label_vars=['descr'], separate=True, keyind=5, ylabel='move_policy_successes', exclude=exclude, split_runs=False, include=include, inter=1200, window=500, fname='tampspeedup', ylim=[(0., 2.5)], xlim=[(0., 12000), (0., 12000), (0., 12000), (0., 10000)]) 
        #gen_data_plots(xvar='time', yvar=['success rate'], keywords=keywords, lab='rollout_info', label_vars=['descr'], separate=True, keyind=5, ylabel='per_goal_success', exclude=exclude, split_runs=False, include=include, inter=10, window=100, fname='pergoal_success', split_goal=True)
        gen_data_plots(xvar='time', yvar=['dagger_success'], keywords=keywords, lab='rollout_info', label_vars=['descr'], separate=True, keyind=5, ylabel='per_goal_success', exclude=exclude, split_runs=False, include=include, inter=10, window=100, fname='dagger_success', split_goal=False)
        #gen_data_plots(xvar='time', yvar=['optimization time', 'plan length', 'opt duration per ts'], keywords=keywords, lab='motion', label_vars=['descr'], separate=True, keyind=5, ylabel='move_policy_successes', exclude=exclude, split_runs=False, include=include, inter=600, window=500, fname='tampspeedup') 
        gen_data_plots(xvar='time', yvar=['episode return'], keywords=keywords, lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='succdataloadtargs', exclude=exclude, split_runs=False, include=include, inter=200, window=100, fname='endreturn_{}'.format(keywords[0]))

        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='control', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='primpol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='primitive', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='primpol')


        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='move_to_grasp_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot4', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='movetograsppol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='lift_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot3', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='liftpol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='slide_open_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot2', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='openpol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='slide_close_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot2', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='closepol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='place_in_door_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot1', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='place_in_pol')
        gen_data_plots(xvar='time', yvar=[['train_component_loss', 'val_component_loss']], keywords=keywords, lab='place_right', label_vars=['descr'], separate=True, keyind=5, ylabel='recentplot1', exclude=exclude, split_runs=True, include=include, inter=60, window=20, fname='placepol')

        if not terminate:
            time.sleep(180)
