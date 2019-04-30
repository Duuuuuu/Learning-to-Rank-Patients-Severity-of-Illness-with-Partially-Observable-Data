import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import argparse
from sklearn.model_selection import train_test_split
from dops import *
from preprocessing import *

# read data
x_train, y_train, bilevel_df, multilevel_df = get_data('./Data/states_bpleq65.p')

# covergae matrix
manual_bilevel_coverage = np.loadtxt('bi_cover.txt', delimiter=',')
manual_multilevel_coverage = np.loadtxt('multi_cover.txt', delimiter=',')

# harder measure
measure_freq = {'bicarbonate': 2.981861,'bun': 1.499739,'creatinine': 1.505387,\
                'fio2': 9.647378,'glucose': 6.117458,'hct': 1.975867,'lactate': 1.330443,\
                'magnesium': 1.955151,'platelets': 1.459454,'potassium': 2.182242,\
                'sodium': 1.744012,'wbc': 1.395242, 'urine': 14.13}
harder_measure = sorted(measure_freq, key=measure_freq.get)

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--savepath", type=str, default='.')
parser.add_argument("--drops", type=int, default=1)
parser.add_argument("--test_size", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=64)
FLAGS = parser.parse_args()

# calculate covered indices
bi_cover, multi_cover = [], []
measure = harder_measure[:FLAGS.drops]
cols = multilevel_df.columns
cover_names = []
for m in measure:
    bi_cover.append(bilevel_df.columns.get_loc(m))
    for i in range(6):
        name = m+str(i)
        if name in cols:
            multi_cover.append(cols.get_loc(name))
            cover_names.append(name)   
        else:
            break
bi_x, multi_x, y = bilevel_df.values, multilevel_df.values, y_train['saps'].values

# init
bi_preds = []
bi_preds_theta = []
multi_preds = []
multi_preds_theta = []
truths = []
print("Starting DOPS when dropping " + str(measure))

# run dops
repeat = 10
t1 = time.time()
for i in range(repeat):
    np.random.seed(i)
    index = np.random.choice(len(y), size=10000, replace=False)
    bi_x_train_dops, multi_x_train_dops, y_train_dops = bi_x[index], multi_x[index], y[index]
    bi_x_train_dops[:, bi_cover] = 0
    multi_x_train_dops[:, multi_cover] = 0
    bi_X_train, bi_X_test,Y_train,Y_test = train_test_split(bi_x_train_dops,y_train_dops,test_size=FLAGS.test_size,random_state=i)
    multi_X_train, multi_X_test,Y_train,Y_test = train_test_split(multi_x_train_dops,y_train_dops,test_size=FLAGS.test_size,random_state=i)
    test_len = len(Y_test)
    
    # bilevel
    bi_res1, bi_pred_theta1, bi_max_item_index = dops(bi_X_train, Y_train, bi_X_test, manual_bilevel_coverage, test_len, 0.8, \
                                           np.zeros(len(manual_bilevel_coverage)), loss='quantile', batch_size=FLAGS.batch_size, eta=1, iters=100, print_every=10,cover_indices=bi_cover)

    # multilevel
    multi_res1, multi_pred_theta1, multi_max_item_index = dops(multi_X_train, Y_train, multi_X_test, manual_multilevel_coverage, test_len, 0.8, \
                                           np.zeros(len(manual_multilevel_coverage)), loss='quantile', batch_size=FLAGS.batch_size, eta=1, iters=100, print_every=10,cover_indices=multi_cover)
    
    # collect information
    bi_preds.append(bi_res1)
    bi_preds_theta.append(bi_pred_theta1)
    multi_preds.append(multi_res1)
    multi_preds_theta.append(multi_pred_theta1)
    truths.append(Y_test)
t2 = time.time()


# save
with open(FLAGS.savepath, 'wb') as f:
    pickle.dump({'time':t2-t1, 'truth':truths, 'multi_preds':multi_preds, 'multi_preds_theta':multi_preds_theta}, f)

