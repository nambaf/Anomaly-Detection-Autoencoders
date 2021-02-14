#!/usr/bin/env python
# coding: utf-8

# # Analisi variabilità con varianza e media. Test con dati termici
# 
# Librerie
import pandas as pd
import numpy as np
from datetime import timedelta 
import matplotlib.pyplot as plt


ter_file = 'PRIVATE DATA - NOT LOADED'
ter_data = pd.read_csv(ter_file)
ter_data.columns
subset = ["PRIVATE DATA - NOT LOADED"]
ter_data = ter_data.loc[:,subset]


# ### Run

ter_data['DATE_TIME'] = pd.to_datetime(ter_data['DATE_TIME'])
for column in ter_data.drop(['DATE_TIME'],axis=1):
    original_var = np.var(ter_data[column])
    print(original_var)

def random_datetimes_or_dates(start, end, out_format='datetime', n=10): 
    (divide_by, unit) = (10**9, 's') if out_format=='datetime' else (24*60*60*10**9, 'D')

    start_u = start.value//divide_by
    end_u = end.value//divide_by

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit) 





random_datetimes_or_dates(pd.to_datetime('2013-10-22'),pd.to_datetime('2014-12-29'), out_format='datetime')





for start_time in random_datetimes_or_dates(pd.to_datetime('2013-10-22'),pd.to_datetime('2014-12-29'), out_format='datetime', n=5):
    end_time = start_time + timedelta(hours = 3)
    subset_ana = ter_data[(ter_data['DATE_TIME']>=start_time) & (ter_data['DATE_TIME']<=end_time)].copy()
    filter_one_min = subset_ana['DATE_TIME'].iloc[::4]
    subset_ana_one_min = pd.merge(subset_ana, filter_one_min, on='DATE_TIME', how='inner')
    filter_ten_min = subset_ana['DATE_TIME'].iloc[::40]
    subset_ana_ten_min = pd.merge(subset_ana, filter_ten_min, on='DATE_TIME', how='inner')
    filter_fiften_min = subset_ana['DATE_TIME'].iloc[::60]
    subset_ana_fiften_min = pd.merge(subset_ana, filter_fiften_min, on='DATE_TIME', how='inner')
    filter_thir_min = subset_ana['DATE_TIME'].iloc[::120]
    subset_ana_thir_min = pd.merge(subset_ana, filter_thir_min, on='DATE_TIME', how='inner')
    
    for column in subset_ana.loc[:,["CHP_HEATPOWER", "CHP-OUT_TEMP", "CHP-IN_TEMP", "CHP_WATER-FLOWRATE", "CHP_GASCONS"]]:
        original_var = np.var(subset_ana[column])
        original_mean = np.mean(subset_ana[column])
        one_min_var = np.var(subset_ana_one_min[column])
        one_min_mean = np.mean(subset_ana_one_min[column])
        ten_min_var = np.var(subset_ana_ten_min[column])
        ten_min_mean = np.mean(subset_ana_ten_min[column])
        fiften_min_var = np.var(subset_ana_fiften_min[column])
        fiften_min_mean = np.mean(subset_ana_fiften_min[column])
        thir_min_var = np.var(subset_ana_thir_min[column])
        thir_min_mean = np.mean(subset_ana_thir_min[column])
        print("-----")
        print("PERIOD start {} and end {}".format(start_time,end_time))
        print("Column {} with original var {:.4f} and mean {:.4f}".format(column,original_var,original_mean))
        print("One min(var,mean): ({:.4f},{:.4f}) - Perc var(%):{:.4f}".format(one_min_var,one_min_mean,one_min_var*100/(original_var if original_var != 0 else 1.0)))
        print("10 min(var,mean): ({:.4f},{:.4f}) - Perc var(%):{:.4f}".format(ten_min_var,ten_min_mean,ten_min_var*100/(original_var if original_var != 0 else 1.0)))
        print("15 min(var,mean): ({:.4f},{:.4f}) - Perc var(%):{:.4f}".format(fiften_min_var,fiften_min_mean,fiften_min_var*100/(original_var if original_var != 0 else 1.0)))
        print("30 min(var,mean): ({:.4f},{:.4f}) - Perc var(%):{:.4f}".format(thir_min_var,thir_min_mean,thir_min_var*100/(original_var if original_var != 0 else 1.0)))
        print("-----")
        # Plot
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)

        ax.plot(subset_ana['DATE_TIME'],subset_ana[column], c='red', alpha=0.2, marker="s", label='original')
        ax.plot(subset_ana['DATE_TIME'].iloc[::4],subset_ana[column].iloc[::4], c='blue', alpha=0.2, marker="s", label='1 min')
        ax.plot(subset_ana['DATE_TIME'].iloc[::40],subset_ana[column].iloc[::40], c='yellow', alpha=0.2, marker="s", label='10 min')
        ax.plot(subset_ana['DATE_TIME'].iloc[::60],subset_ana[column].iloc[::60], c='green', alpha=0.2, marker="s", label='15 min')
        ax.plot(subset_ana['DATE_TIME'].iloc[::120],subset_ana[column].iloc[::120], c='black', alpha=0.2, marker="s", label='30 min')
        
        plt.legend(loc='upper left');
        plt.title('Column {}, period {}---{}'.format(column,start_time,end_time))
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()







