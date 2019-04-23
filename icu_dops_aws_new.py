import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from dops import *
from itertools import combinations


if __name__ == "__main__":

    with open('./Data/states_bpleq65.p','rb') as f:
        states = pickle.load(f)

    patient_ids = list(states.keys())


    pid = patient_ids[0]
    df_clean = states[pid].loc[:,~states[pid].columns.isin(['Times','bicarbonate_ind', 'bun_ind','creatinine_ind',
                                                            'fio2_ind','glucose_ind','hct_ind','hr_ind','lactate_ind',
                                                            'magnesium_ind','meanbp_ind','platelets_ind','potassium_ind',
                                                            'sodium_ind','spo2_ind','spontaneousrr_ind','temp_ind',
                                                            'urine_ind','wbc_ind'])]

    df_clean.insert(0,'ICU_ID',pid)
    df_x_train = df_clean.loc[:,~df_clean.columns.isin(['sofa','oasis','saps'])]
    df_y_train = df_clean[['sofa','oasis','saps']]

    x_train = df_x_train.iloc[0]
    y_train = df_y_train.iloc[0]

    for pid in patient_ids[1:]:
        df_clean = states[pid].loc[:,~states[pid].columns.isin(['Times','bicarbonate_ind', 'bun_ind','creatinine_ind',
                                                                'fio2_ind','glucose_ind','hct_ind','hr_ind','lactate_ind',
                                                                'magnesium_ind','meanbp_ind','platelets_ind','potassium_ind',
                                                                'sodium_ind','spo2_ind','spontaneousrr_ind','temp_ind',
                                                                'urine_ind','wbc_ind'])]

        df_clean.insert(0,'ICU_ID',pid)
        df_x_train = df_clean.loc[:,~df_clean.columns.isin(['sofa','oasis','saps'])]
        df_y_train = df_clean[['sofa','oasis','saps']]

        x_train = pd.concat([x_train,df_x_train.iloc[0]],axis=1)
        y_train = pd.concat([y_train,df_y_train.iloc[0]],axis=1)

    x_train = x_train.T.reset_index(drop=True)
    y_train = y_train.T.reset_index(drop=True)
    x_train = x_train.drop(['ICU_ID'],axis=1)

    ##Multilevel Transformation
    multilevel_header = ["age1","age2","age3","age4","age5","is_F",
                         "weight","surg_ICU","is_not_white","is_emergency","is_urgent",
                         "hrs_from_admit_to_icu","bicarbonate1","bicarbonate2","bun","creatinine",
                         "fio2","glucose","hct","hr1","hr2","hr3","hr4","lactate","magnesium","meanbp",
                         "platelets","potassium1","potassium2","sodium1","sodium2","spo2","spontaneousrr","temp","urine1","urine2","wbc"]

    multilevel_df = pd.DataFrame(0,index=np.arange(len(x_train)),columns=multilevel_header)

    for i,row in x_train.iterrows():
        if 40 < row['age'] <= 59:
            multilevel_df.iloc[i]['age1'] = 1
        elif 60 < row['age'] <= 69:
            multilevel_df.iloc[i]['age2'] = 1
        elif 70 < row['age'] <= 74:
            multilevel_df.iloc[i]['age3'] = 1
        elif 75 < row['age'] <= 79:
            multilevel_df.iloc[i]['age4'] = 1
        elif row['age'] > 80:
            multilevel_df.iloc[i]['age5'] = 1

        multilevel_df.iloc[i]['is_F'] = 1 if row['is_F'] == 1 else 0
        multilevel_df.iloc[i]['weight'] = 1 if row['weight'] >= 75 else 0
        multilevel_df.iloc[i]['hrs_from_admit_to_icu'] = 1 if row['hrs_from_admit_to_icu'] >= 1 else 0

        if 15 <= row['bicarbonate'] <= 19:
            multilevel_df.iloc[i]['bicarbonate1'] = 1
        elif row['bicarbonate'] < 15:
            multilevel_df.iloc[i]['bicarbonate2'] = 1

        multilevel_df.iloc[i]['bun'] = 0 if 7 <= row['bun'] <= 12 else 1
        multilevel_df.iloc[i]['creatinine'] = 0 if 0.5 <= row['creatinine'] <= 1.2 else 1
        multilevel_df.iloc[i]['fio2'] = 1 if row['fio2'] >= 0.5 else 0
        multilevel_df.iloc[i]['glucose'] = 1 if row['glucose'] >= 125 else 0
        multilevel_df.iloc[i]['hct'] = 0 if 37 <= row['hct'] <= 52 else 1

        if 40 <= row['hr'] <= 69:
            multilevel_df.iloc[i]['hr1'] = 1
        elif 120 <= row['hr'] <= 159:
            multilevel_df.iloc[i]['hr2'] = 1
        elif row['hr'] >= 160:
            multilevel_df.iloc[i]['hr3'] = 1
        elif row['hr'] < 40:
            multilevel_df.iloc[i]['hr4'] = 1

        multilevel_df.iloc[i]['lactate'] = 1 if row['lactate'] >= 2 else 0
        multilevel_df.iloc[i]['magnesium'] = 0 if 1.5 <= row['magnesium'] <= 2.5 else 1
        multilevel_df.iloc[i]['meanbp'] = 1 if row['meanbp'] <= 65 else 0
        multilevel_df.iloc[i]['platelets'] = 0 if 140 <= row['platelets'] <= 450 else 1

        if row['potassium'] >= 5:
            multilevel_df.iloc[i]['potassium1'] = 1
        elif row['potassium'] < 3:
            multilevel_df.iloc[i]['potassium2'] = 1

        if row['sodium'] >= 145:
            multilevel_df.iloc[i]['sodium1'] = 1
        elif row['sodium'] < 125:
            multilevel_df.iloc[i]['sodium2'] = 1

        multilevel_df.iloc[i]['spo2'] = 1 if row['spo2'] <= 95 else 0
        multilevel_df.iloc[i]['spontaneousrr'] = 0 if 12 <= row['spontaneousrr'] <= 25 else 1
        multilevel_df.iloc[i]['temp'] = 1 if row['temp'] >= 39 else 0

        if 50 <= row['urine'] <= 99:
            multilevel_df.iloc[i]['urine1'] = 1
        elif row['urine'] < 50:
            multilevel_df.iloc[i]['urine2'] = 1

        multilevel_df.iloc[i]['wbc'] = 0 if 4.3 <= row['wbc'] <= 10.8 else 1

    manual_multilevel_coverage = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,1,0,1,0,1],
                                         [0,0,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,0,1,1],
                                         [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,0,0,1,0,1],
                                         [0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0],
                                         [0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],
                                         [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1],
                                         [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,1,0,1],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
                                         [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
                                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1]])

    ## bilevel_df Transformation
    ##1's indicate the patient's feature is in bad condition, 0's means feature values in normal

    bilevel_df = x_train.copy()

    bilevel_df['age'] = np.where(bilevel_df['age']>=60,1,0)
    bilevel_df['weight'] = np.where(bilevel_df['weight']>=75,1,0)
    bilevel_df['hrs_from_admit_to_icu'] = np.where(bilevel_df['hrs_from_admit_to_icu']>=1,1,0)
    bilevel_df['bicarbonate'] = np.where(bilevel_df['bicarbonate']<=20,1,0)
    bilevel_df['bun'] = np.where(np.logical_and(bilevel_df['bun']>=7,bilevel_df['bun']<=20),0,1)
    bilevel_df['creatinine'] = np.where(np.logical_and(bilevel_df['creatinine']>=0.5,bilevel_df['creatinine']<=1.2),0,1)
    bilevel_df['fio2'] = np.where(bilevel_df['fio2']>=0.5,1,0)
    bilevel_df['glucose'] = np.where(bilevel_df['glucose']>=125,1,0)
    bilevel_df['hct'] = np.where(np.logical_and(bilevel_df['hct']>=37,bilevel_df['hct']<=52),0,1)
    bilevel_df['hr'] = np.where(np.logical_and(bilevel_df['hr']>=70,bilevel_df['hr']<=119),0,1)
    bilevel_df['lactate'] = np.where(bilevel_df['lactate']>=2,1,0)
    bilevel_df['magnesium'] = np.where(np.logical_and(bilevel_df['magnesium']>=1.5,bilevel_df['magnesium']<=2.5),0,1)
    bilevel_df['meanbp'] = np.where(bilevel_df['meanbp']<=65,1,0)
    bilevel_df['platelets'] = np.where(np.logical_and(bilevel_df['platelets']>=140,bilevel_df['platelets']<=450),0,1)
    bilevel_df['potassium'] = np.where(np.logical_and(bilevel_df['potassium']>=3,bilevel_df['potassium']<=4.9),0,1)
    bilevel_df['sodium'] = np.where(np.logical_and(bilevel_df['sodium']>=125,bilevel_df['potassium']<=144),0,1)
    bilevel_df['spo2'] = np.where(bilevel_df['spo2']<=95,1,0)
    bilevel_df['spontaneousrr'] = np.where(np.logical_and(bilevel_df['spontaneousrr']>=12,bilevel_df['spontaneousrr']<=25),0,1)
    bilevel_df['temp'] = np.where(bilevel_df['temp']>=39,1,0)
    bilevel_df['urine'] = np.where(bilevel_df['urine']<=100,1,0)
    bilevel_df['wbc'] = np.where(np.logical_and(bilevel_df['wbc']>=4.3,bilevel_df['wbc']<=10.8),0,1)

    manual_bilevel_coverage = np.array([[1,0,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1],
    [0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
    [1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0],
    [1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0],
    [1,0,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1]])

    harder_measure = ['lactate','platelets','bun','hct','creatinine','magnesium','potassium','bicarbonate','sodium']

    bilevel_mti = {
    'lactate':[15],
    'platelets':[18],
    'bun':[9],
    'hct':[13],
    'creatinine':[10],
    'magnesium':[16],
    'potassium':[19],
    'bicarbonate':[8],
    'sodium':[20]
    }

    multilevel_mti = {
        'lactate':[23],
        'platelets':[26],
        'bun':[14],
        'hct':[18],
        'creatinine':[15],
        'magnesium':[24],
        'potassium':[27,28],
        'bicarbonate':[12,13],
        'sodium':[29,30],
    }

    # Randomly select 1000 items from the master
    #for item in combinations(harder_measure,2):
    for i in range(1):

        #measure = list(item)
        measure = harder_measure
        print("Starting DOPS when dropping " + str(measure))

        bi_cover,multi_cover = [],[]
        for m in measure:
            bi_cover.extend(bilevel_mti[m])
            multi_cover.extend(multilevel_mti[m])

        bi_preds = []
        bi_preds_cover_self = []
        multi_preds = []
        multi_preds_cover_self = []
        truths = []
        t1 = time.time()
        repeat = 10
        for i in range(repeat):
            start = time.time()
            np.random.seed(i)
            index = np.random.choice(len(multilevel_df), size=1000, replace=False)
            bi_x, multi_x, y = bilevel_df.iloc[index], multilevel_df.iloc[index], y_train.iloc[index]
            bi_x[measure] = 0.0
            multi_x[measure] = 0.0
            bi_x_train_dops = bi_x.values
            multi_x_train_dops = multi_x.values
            y_train_dops = y['saps'].values + y['oasis'].values + y['sofa'].values
            bi_X_train, bi_X_test,Y_train,Y_test = train_test_split(bi_x_train_dops,y_train_dops,test_size=0.02,random_state=i)
            multi_X_train, multi_X_test,Y_train,Y_test = train_test_split(multi_x_train_dops,y_train_dops,test_size=0.02,random_state=i)
            test_len = len(Y_test)
            bi_res1, bi_pred_theta, bi_max_item_index = dops(bi_X_train, Y_train, bi_X_test, manual_bilevel_coverage, test_len, 0.8, \
                                                   np.zeros(26), batch_size=64, eta=1, iters=200, print_every=40,cover_indices=bi_cover)
            bi_res2, bi_pred_theta, bi_max_item_index = dops(bi_X_train, Y_train, bi_X_test, np.eye(len(manual_bilevel_coverage)), test_len, 0.8, \
                                                   np.zeros(26), batch_size=64, eta=1, iters=200, print_every=40,cover_indices=bi_cover)

            multi_res1, multi_pred_theta, multi_max_item_index = dops(multi_X_train, Y_train, multi_X_test, manual_multilevel_coverage, test_len, 0.8, \
                                                   np.zeros(37), batch_size=64, eta=1, iters=200, print_every=40,cover_indices=multi_cover)
            multi_res2, multi_pred_theta, multi_max_item_index = dops(multi_X_train, Y_train, multi_X_test, np.eye(len(manual_multilevel_coverage)), test_len, 0.8, \
                                                   np.zeros(37), batch_size=64, eta=1, iters=200, print_every=40,cover_indices=multi_cover)

            bi_preds.append(bi_res1)
            bi_preds_cover_self.append(bi_res2)
            multi_preds.append(multi_res1)
            multi_preds_cover_self.append(multi_res2)
            truths.append(Y_test)
            end = time.time()
            print("One repetition takes {0} seconds".format(end-start))

        bi_pred = []
        bi_pred_cover_self = []
        multi_pred = []
        multi_pred_cover_self = []
        real = []
        rand = []
        for i in range(repeat):
            bi_pred.append(truths[i][np.argmax(bi_preds[i])])
            bi_pred_cover_self.append(truths[i][np.argmax(bi_preds_cover_self[i])])
            multi_pred.append(truths[i][np.argmax(multi_preds[i])])
            multi_pred_cover_self.append(truths[i][np.argmax(multi_preds_cover_self[i])])
            real.append(max(truths[i]))
            rand.append(np.median(truths[i]))
        t2 = time.time()
        print('Used time: %.2fs' % (t2-t1))

        plt.figure(figsize=(12,4))
        plt.plot(np.arange(0,repeat), bi_pred, '*-', alpha=0.5, label='Bilevel C(S)')
        plt.plot(np.arange(0,repeat), bi_pred_cover_self, '*-', alpha=0.5, label='Bilevel S')
        plt.plot(np.arange(0,repeat), multi_pred, '*-', alpha=0.5, label='Multilevel C(S)')
        plt.plot(np.arange(0,repeat), multi_pred_cover_self, '*-', alpha=0.5, label='Multilevel S')
        plt.plot(np.arange(0,repeat), rand, '*-', alpha=0.5, label='median')
        plt.plot(np.arange(0,repeat), real, '*-', alpha=0.5, label='ground truth')
        plt.legend()
        plt.title('Multilevel Maximum items with DOPS algo and sampling randomly after dropping ' + str(measure))
        plt.savefig("Multilevel results after dropping " + str(measure))
