import numpy as np
import pandas as pd
import seaborn as sns
from plotting import plot
import matplotlib.pyplot as plt

data = open('../experiment_logs/namo_objs1_1/olivia_train_6/rollout_logs/MotionServer12_verbose.txt', "r")
data2 = open('../experiment_logs/namo_objs1_1/olivia_train_6/policy_control_log.txt, "r")
data = eval(data).read().split("\n\n")
values = []

for each_list in data:
    values.append(each_list[-1].values())

#if value = 1
#for i in data:
    #if i['value'] == 1.0:
    #list of dictionaries
#    opt_success = data['opt_success']


#training and val loss need to be going down
""" train_loss = []
val_loss = []
time = []
N = []
for i in data2:
    #y values 
    train_loss.append(i['train_loss'])
    val_loss.append(i['val_loss'])
    #x values
    time.append(i['time'])
    N.append(i['N'])
    torch_iter.append(i['torch_iter']) """
