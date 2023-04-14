import numpy as np
import pandas as pd
import seaborn as sns
from plotting import plot

data = np.load('../experiment_logs/namo_objs1_1/mac_test_2/test_hl_test_log.npy')
i = 2
data_dict = {}
data_list = []
for pts in data:
    pt = pts[0]
    no, nt = int(pt[4]), int(pt[5])
    #if label.find('cereal-and-milk') >= 0 and pt[10] > 1400: continue
    #pt[1] = pt[1] * 20 # Path len
    data_list.append({'time': pt[3], 'success at end': pt[0], 'path length': 20. * pt[1], 'distance from goal': pt[2], 'n_data': pt[6], 'key': (no, nt), 'ind': i, 'success anywhere': pt[7], 'optimal_rollout_success': pt[9], 'number of plans': pt[10], 'subgoals anywhere': pt[11], 'subgoals closest distance': pt[12], 'collision': pt[8], 'exp id': i})
    # all_data[k][full_exp][cur_dir][cur_dir].append({'time': pt[3], 'success at end': pt[0], 'path length': 20. * pt[1], 'distance from goal': pt[2], 'n_data': pt[6], 'key': (no, nt), 'label': label, 'ind': i, 'success anywhere': pt[7], 'optimal_rollout_success': pt[9], 'number of plans': pt[10], 'subgoals anywhere': pt[11], 'subgoals closest distance': pt[12], 'collision': pt[8], 'exp id': i})
    if len(pt) > 13:
        data_list[-1]['any target'] = pt[13]
    if len(pt) > 14:
        data_list[-1]['smallest tolerance'] = pt[14]
    if len(pt) > 16:
        data_list[-1]['success with postcond'] = pt[16]
        #all_data[k][full_exp][cur_dir][cur_dir][-1]['success with postcond'] = pt[16]
    if len(pt) > 17:
        data_list[-1]['success with adj_eta'] = pt[17]
        # all_data[k][full_exp][cur_dir][cur_dir][-1]['success with adj_eta'] = pt[17]
    if len(pt) > 18:
        data_list[-1]['episode return'] = pt[18]
        # all_data[k][full_exp][cur_dir][cur_dir][-1]['episode return'] = pt[18]
    if len(pt) > 19:
        data_list[-1]['episode reward'] = pt[19]
        # all_data[k][full_exp][cur_dir][cur_dir][-1]['episode reward'] = pt[19]
df = pd.DataFrame.from_records(data_dict, columns=["time", "episode return"])
df.set_index('time')
print(df.head())

#sns.set()
#sns.relplot()