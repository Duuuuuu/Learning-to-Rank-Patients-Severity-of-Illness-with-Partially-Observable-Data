import pickle
import numpy as np
import pandas as pd


def get_data(path):
    # -------------------- read ----------------------
    with open('./Data/states_bpleq65.p','rb') as f:
        states = pickle.load(f)
    patient_ids = list(states.keys())


    # -------------------- clean ---------------------
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


    # ----------- encoding binary level ------------
    # 1's indicate the patient's feature is in bad condition, 0's means feature values in normal
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


    # ----------- encoding multilevel ----------------
    # encode multilevel (one hot encoding)
    # "[measure]0 = 1" means normal condition
    multilevel_header = ["age0","age1","age2","age3","age4","age5","is_F0", "is_F1", "weight0", "weight1", 
                         "surg_ICU0","surg_ICU1","is_not_white0","is_not_white1","is_emergency0","is_emergency1",
                         "is_urgent0","is_urgent1","hrs_from_admit_to_icu0","hrs_from_admit_to_icu1",
                         "bicarbonate0","bicarbonate1","bicarbonate2","bun0","bun1","creatinine0","creatinine1",
                         "fio20","fio21","glucose0","glucose1","hct0","hct1","hr0","hr1","hr2","hr3","hr4",
                         "lactate0","lactate1","magnesium0","magnesium1","meanbp0","meanbp1","platelets0","platelets1",
                         "potassium0","potassium1","potassium2","sodium0","sodium1","sodium2","spo20","spo21",
                         "spontaneousrr0","spontaneousrr1","temp0","temp1","urine0","urine1","urine2","wbc0","wbc1"]

    multilevel_df = pd.DataFrame(0,index=np.arange(len(x_train)),columns=multilevel_header)
    for i,row in x_train.iterrows():
        # age
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
        else:
            multilevel_df.iloc[i]['age0'] = 1 
        
        # is_F
        if row['is_F'] == 1:
            multilevel_df.iloc[i]['is_F1'] = 1
        else:
            multilevel_df.iloc[i]['is_F0'] = 1
        
        # is_not_white
        if row['is_not_white'] == 1:
            multilevel_df.iloc[i]['is_not_white1'] = 1
        else:
            multilevel_df.iloc[i]['is_not_white0'] = 1
        
        # is_emergency
        if row['is_not_white'] == 1:
            multilevel_df.iloc[i]['is_emergency1'] = 1
        else:
            multilevel_df.iloc[i]['is_emergency0'] = 1
            
        # is_urgent
        if row['is_urgent'] == 1:
            multilevel_df.iloc[i]['is_urgent1'] = 1
        else:
            multilevel_df.iloc[i]['is_urgent0'] = 1
            
        # weight
        if row['weight'] >= 75:
            multilevel_df.iloc[i]['weight1'] = 1
        else:
            multilevel_df.iloc[i]['weight0'] = 1
            
        # surg_ICU
        if row['surg_ICU'] == 1:
            multilevel_df.iloc[i]['surg_ICU1'] = 1
        else:
            multilevel_df.iloc[i]['surg_ICU0'] = 1
        
        # hrs_from_admit_to_icu
        if row['hrs_from_admit_to_icu'] >= 1:
            multilevel_df.iloc[i]['hrs_from_admit_to_icu1'] = 1
        else:
            multilevel_df.iloc[i]['hrs_from_admit_to_icu0'] = 1
        
        # bicarbonate
        if 15 <= row['bicarbonate'] <= 19:
            multilevel_df.iloc[i]['bicarbonate1'] = 1
        elif row['bicarbonate'] < 15:
            multilevel_df.iloc[i]['bicarbonate2'] = 1
        else:
            multilevel_df.iloc[i]['bicarbonate0'] = 1
        
        # bun
        if 7 <= row['bun'] <= 12:
            multilevel_df.iloc[i]['bun0'] = 1
        else:
            multilevel_df.iloc[i]['bun1'] = 1
        
        # creatinine
        if 0.5 <= row['creatinine'] <= 1.2:
            multilevel_df.iloc[i]['creatinine0'] = 1
        else:
            multilevel_df.iloc[i]['creatinine1'] = 1
        
        # fio2
        if row['fio2'] >= 0.5:
            multilevel_df.iloc[i]['fio21'] = 1 
        else:
            multilevel_df.iloc[i]['fio20'] = 1 
        
        # glucose
        if row['glucose'] >= 125:
            multilevel_df.iloc[i]['glucose1'] = 1
        else:
            multilevel_df.iloc[i]['glucose0'] = 1
        
        # hct
        if 37 <= row['hct'] <= 52:
            multilevel_df.iloc[i]['hct0'] = 1
        else:
            multilevel_df.iloc[i]['hct1'] = 1
        
        # hr
        if 40 <= row['hr'] <= 69:
            multilevel_df.iloc[i]['hr1'] = 1
        elif 120 <= row['hr'] <= 159:
            multilevel_df.iloc[i]['hr2'] = 1
        elif row['hr'] >= 160:
            multilevel_df.iloc[i]['hr3'] = 1
        elif row['hr'] < 40:
            multilevel_df.iloc[i]['hr4'] = 1
        else:
            multilevel_df.iloc[i]['hr0'] = 1
        
        # lactate
        if row['lactate'] >= 2:
            multilevel_df.iloc[i]['lactate1'] = 1
        else:
            multilevel_df.iloc[i]['lactate0'] = 1
        
        # magnesium
        if 1.5 <= row['magnesium'] <= 2.5:
            multilevel_df.iloc[i]['magnesium0'] = 1
        else:
            multilevel_df.iloc[i]['magnesium1'] = 1
            
        # meanbp
        if row['meanbp'] <= 65:
            multilevel_df.iloc[i]['meanbp1'] = 1 
        else:
            multilevel_df.iloc[i]['meanbp0'] = 1 
        
        # platelets
        if 140 <= row['platelets'] <= 450:
            multilevel_df.iloc[i]['platelets0'] = 1
        else:
            multilevel_df.iloc[i]['platelets1'] = 1
        
        # potassium
        if row['potassium'] >= 5:
            multilevel_df.iloc[i]['potassium1'] = 1
        elif row['potassium'] < 3:
            multilevel_df.iloc[i]['potassium2'] = 1
        else:
            multilevel_df.iloc[i]['potassium0'] = 0
        
        # sodium
        if row['sodium'] >= 145:
            multilevel_df.iloc[i]['sodium1'] = 1
        elif row['sodium'] < 125:
            multilevel_df.iloc[i]['sodium2'] = 1
        else:
            multilevel_df.iloc[i]['sodium0'] = 1
        
        # spo2
        if row['spo2'] <= 95:
            multilevel_df.iloc[i]['spo21'] = 1
        else:
            multilevel_df.iloc[i]['spo20'] = 1
        
        # spontaneousrr
        if 12 <= row['spontaneousrr'] <= 25:
            multilevel_df.iloc[i]['spontaneousrr0'] = 1
        else:
            multilevel_df.iloc[i]['spontaneousrr1'] = 1
        
        # temp
        if row['temp'] >= 39:
            multilevel_df.iloc[i]['temp1'] = 1
        else:
            multilevel_df.iloc[i]['temp0'] = 1
        
        # urine
        if 50 <= row['urine'] <= 99:
            multilevel_df.iloc[i]['urine1'] = 1
        elif row['urine'] < 50:
            multilevel_df.iloc[i]['urine2'] = 1
        else:
            multilevel_df.iloc[i]['urine0'] = 1
            
        # wbc
        if 4.3 <= row['wbc'] <= 10.8:
            multilevel_df.iloc[i]['wbc0'] = 1
        else:
            multilevel_df.iloc[i]['wbc1'] = 1


    return x_train, y_train, bilevel_df, multilevel_df



