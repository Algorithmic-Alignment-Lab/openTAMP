import numpy as np
import pandas as pd
import seaborn as sns
from plotting import plot
import matplotlib.pyplot as plt

data = open('../experiment_logs/namo_objs1_1/olivia_train_6/rollout_logs/MotionServer12_verbose.txt', "r")
data = data.read()
data = str.split(data, '\n\n')
value_list = []
for each_list in data:
    try:
        print(each_list)
        each_list = eval(each_list)
        value_list.append(each_list[-1].values())
    except:
        continue
print(value_list)
#if value = 1
#for i in data:
    #if i['value'] == 1.0:
    #list of dictionaries
#    opt_success = data['opt_success']

#data2 = open('../experiment_logs/namo_objs1_1/olivia_train_6/policy_control_log.txt, "r")


#training and val loss need to be going down
#train_loss = []
#val_loss = []
#time = []
#N = []
#for i in data2:
    #y values 
 #   train_loss.append(i['train_loss'])
  #  val_loss.append(i['val_loss'])
    #x values
   # time.append(i['time'])
   # N.append(i['N'])
   # torch_iter.append(i['torch_iter']) """
