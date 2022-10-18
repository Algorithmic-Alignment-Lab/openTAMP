import imp
import importlib
import pickle
import os
import os.path
import random
import shutil
import time


LOG_DIR = 'experiment_logs/'
MODEL_DIR = 'torch_saved/'


def load_config(args, config=None, reload_module=None):
    config_file = args.config
    if reload_module is not None:
        config_module = reload_module
        imp.reload(config_module)
    else:
        config_module = importlib.import_module(config_file)
    config = config_module.refresh_config(args.nobjs, args.nobjs)
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    config['num_targs'] = args.ntargs if args.nobjs > 0 else config['num_targs']
    config['server_id'] = args.server_id if args.server_id != '' else str(random.randint(0,2**32))
    config['descr'] = config.get('descr', args.descr)
    dir_name = config['base_weight_dir'] + '/objs{0}_{1}/{2}'.format(config['num_objs'], config['num_targs'], config['descr'])
    config['weight_dir'] = dir_name 

    return config, config_module

def setup_dirs(c, args, rank=0):
    current_id = 0 if c.get('index', -1) < 0 else c['index']

    if c.get('index', -1) < 0:
        while os.path.isdir(LOG_DIR+c['weight_dir']+'_'+str(current_id)):
            current_id += 1

    c['group_id'] = current_id
    c['weight_dir'] = c['weight_dir']+'_{0}'.format(current_id)
    dir_name = ''
    dir_name2 = ''
    sub_dirs = [LOG_DIR] + c['weight_dir'].split('/') + ['rollout_logs']
    sub_dirs2 = [MODEL_DIR] + c['weight_dir'].split('/') + ['rollout_logs']

    c['rank'] = rank
    if rank == 0:
        for d_ind, d in enumerate(sub_dirs):
            dir_name += d + '/'
            dir_name2 += sub_dirs2[d_ind] + '/'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            if not os.path.isdir(dir_name2):
                os.mkdir(dir_name2)

        if not os.path.isdir(dir_name+'samples/'):
            os.mkdir(dir_name+'samples/')

        if args.hl_retrain:
            src = LOG_DIR + args.hl_data + '/hyp.py'

        elif hasattr(args, 'expert_path') and len(args.expert_path):
            src = args.expert_path+'/hyp.py'

        elif len(args.test):
            src = LOG_DIR + '/' + args.test + '/hyp.py'

        else:
            src = c['source'].replace('.', '/')+'.py'

        shutil.copyfile(src, LOG_DIR+c['weight_dir']+'/hyp.py')
        with open(LOG_DIR+c['weight_dir']+'/__init__.py', 'w+') as f:
            f.write('')

        with open(LOG_DIR+c['weight_dir']+'/args.pkl', 'wb+') as f:
            pickle.dump(args, f, protocol=0)

        with open(LOG_DIR+c['weight_dir']+'/args.txt', 'w+') as f:
            f.write(str(vars(args)))

    else:
        time.sleep(0.1) # Give others a chance to let base set up dirs

    return current_id


def check_dirs(config):
    if not os.path.exists(LOG_DIR+config['weight_dir']):
        os.makedirs(LOG_DIR+config['weight_dir'])
    if not os.path.exists(MODEL_DIR+config['weight_dir']):
        os.makedirs(MODEL_DIR+config['weight_dir'])


def get_dir_name(base, no, nt, ind, descr, args=None):
    dir_name = base + 'objs{0}_{1}/{2}'.format(no, nt, descr)
    return dir_name

