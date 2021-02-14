#!/usr/bin/env python
# coding: utf-8

# ## Analisi del PRIVATE DATA - NOT LOADED

# Per lo studio dell'impianto PRIVATE DATA - NOT LOADED utilizzare entrambi i dataset PRIVATE DATA - NOT LOADED. Nessun cambiamento nelle curve di comportamento normale

# Librerie:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# File di riferimento
PRIVATE DATA - NOT LOADED = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'
PRIVATE DATA - NOT LOADED = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'

# File generati
# Cartella di riferimento per le reti neurali PRIVATE DATA - NOT LOADED
output_file_artf_pre_filter = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'
output_file_artf_post_filter = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'

output_file_artf_normal_beh = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'
output_file_artf_anom_beh = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'




PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED)
PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED)
PRIVATE DATA - NOT LOADED = pd.concat([PRIVATE DATA - NOT LOADED, PRIVATE DATA - NOT LOADED], ignore_index=True, sort=False)



# I primi 5 elementi
PRIVATE DATA - NOT LOADED.head(5)




print('Dimensionalità dataset PRIVATE DATA - NOT LOADED:',PRIVATE DATA - NOT LOADED.shape)


# # Individuazione comportamento normale



# Libreria
from lib_preprocessing import calc_gradient, calc_gradient_avg, PRIVATE DATA - NOT LOADED, PRIVATE DATA - NOT LOADED, connectpoints, check_belong_to_curve


# **Aggiunta gradiente per ogni attributo**


ratio = 1 # 15 secondi, ossia il precedente
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED DATA - NOT PRIVATE DATA - NOT LOADED', ratio), axis=1)
ratio = 4 # 1 minuto, valori mediati
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient_avg(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED-PRIVATE DATA - NOT LOADED', ratio), axis=1)

ratio = 1 # 15 secondi, ossia il precedente
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED', ratio), axis=1)
ratio = 4 # 1 minuto, valori mediati
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient_avg(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED', ratio), axis=1)

ratio = 1 # 15 secondi, ossia il precedente
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED', ratio), axis=1)
ratio = 4 # 1 minuto, valori mediati
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(lambda x: calc_gradient_avg(PRIVATE DATA - NOT LOADED, x, 'PRIVATE DATA - NOT LOADED', ratio), axis=1)


# **Aggiunta variabile carico**


PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(chp_creation_ele_perf_ratio,_maxvalue_actpower=(0), axis=1)
PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED.apply(chp_creation_ter_perf_ratio, axis=1)


# Salvataggio Dataframe:



# Salvataggio
PRIVATE DATA - NOT LOADED.to_csv(PRIVATE DATA - NOT LOADED,index=False)


# **Studio preliminare comportamento normale - 20%**


PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED)



### Curve normal beh.
PRIVATE DATA - NOT LOADED=np.array([0, 25, 50, 75, 100])
PRIVATE DATA - NOT LOADED=np.array([0, 20.5, 26, 32.1, 38.4])
PRIVATE DATA - NOT LOADED=np.array([0, 79.8, 79.8, 80.0, 79.9])
PRIVATE DATA - NOT LOADED=np.array([0, 82.5, 83.3, 84.4, 85.0])
PRIVATE DATA - NOT LOADED=np.array([6.588, 6.588, 6.588, 6.588, 6.594])
PRIVATE DATA - NOT LOADED=np.array([0, 3.3398, 4.7549, 6.2635, 7.7514])
K = 0.20
# Upperbound
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1+K)
PRIVATE DATA - NOT LOADED[0] = 20
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1+K)
PRIVATE DATA - NOT LOADED[0] = 55
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1+K)
PRIVATE DATA - NOT LOADED[0] = 55
PRIVATE DATA - NOT LOADED=water_flow_params*(1+K)
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1+K)
PRIVATE DATA - NOT LOADED[0] = 4
# Lowerbound
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1-K)
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1-K)
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1-K)
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1-K)
PRIVATE DATA - NOT LOADED=PRIVATE DATA - NOT LOADED*(1-K)




PRIVATE DATA - NOT LOADED = ['PRIVATE DATA - NOT LOADED']

for index_col, value_col in enumerate(PRIVATE DATA - NOT LOADED):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    if index_col == 0:
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, c='red', label='Upper bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED * 0.9, c='orange', label='Lower bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, 'go', label='Normal beh.')
        connectpoints(ax, load_ratio_params,PRIVATE DATA - NOT LOADED)
    if index_col == 1:
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, c='red', label='Upper bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED * 0.9, c='orange', label='Lower bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, 'go', label='Normal beh.')
        connectpoints(ax, load_ratio_params,PRIVATE DATA - NOT LOADED)
    if index_col == 2:
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, c='red', label='Upper bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED * 0.9, c='orange', label='Lower bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, 'go', label='Normal beh.')
        connectpoints(ax, load_ratio_params,PRIVATE DATA - NOT LOADED)
    if index_col == 3:
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, c='red', label='Upper bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED * 0.9, c='orange', label='Lower bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, 'go', label='Normal beh.')
        connectpoints(ax, load_ratio_params,PRIVATE DATA - NOT LOADED)
    if index_col == 4:
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, c='red', label='Upper bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED * 0.9, c='orange', label='Lower bound 10 %')
        ax.plot(load_ratio_params, PRIVATE DATA - NOT LOADED, 'go', label='Normal beh.')
        connectpoints(ax, load_ratio_params,PRIVATE DATA - NOT LOADED)
        
        
    ax.scatter(PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'], PRIVATE DATA - NOT LOADED[value_col], c='grey', alpha=0.2, marker="s", label='Real data')
    
    plt.legend(loc='upper left');
    plt.title('{} normal behav.'.format(value_col))
    plt.xlabel('Load ratio')
    plt.grid()
    plt.ylabel('{} value'.format(value_col))
    plt.show()


# ELIMINAZIONE entries sotto carico. Carico utile da 10% a 100%. 2.5kw potenza minima per accensione macchina(ma sensori accesi)



PRIVATE DATA - NOT LOADED = PRIVATE DATA - NOT LOADED[PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] >= 10]


# ELIMINAZIONE entries gas consumato con carico nullo. Per training set da togliere ma aggiungere nel test set come anomalia


PRIVATE DATA - NOT LOADED = PRIVATE DATA - NOT LOADED[(PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] > 0) & (PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] != 0)]


# **Feature PRIVATE DATA - NOT LOADED**


PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'] = PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED-PRIVATE DATA - NOT LOADED'] - PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED-PRIVATE DATA - NOT LOADED']


# dopo meeting 25/10/19 concordato curve differenza temperature(1.5 con carico 10%,5 con carico 100% ma range più alti)


# Salvataggio Dataframe:



# Salvataggio
PRIVATE DATA - NOT LOADED.to_csv(output_file_artf_post_filter,index=False)


# ### Creazione dataframe di comportamento normale attraverso le curve trovate


PRIVATE DATA - NOT LOADED = pd.read_csv(output_file_artf_post_filter)
PRIVATE DATA - NOT LOADED['DATE_TIME'] = pd.to_datetime(PRIVATE DATA - NOT LOADED['DATE_TIME'])


# **Studio gradiente PRIVATE DATA - NOT LOADED**


offset_low = .5
offset = 10
sigma1 = 10
tmax = PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'].max()

levels = np.concatenate((
    np.linspace(0, PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'].mean(), 10),
    np.linspace(PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'].mean() + offset_low, 1, 2),
    np.linspace(1 + offset_low, PRIVATE DATA - NOT LOADED['PRIVATE DATA - NOT LOADED'].mean() + offset * sigma1, 2),
    [tmax]
    ))

levels = levels[levels <= tmax]
levels.sort()

class nlcmap(object):
    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(0.0, self.levmax,
             len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)
    
cmap_nonlin = nlcmap(plt.cm.jet, levels)




# Salvataggio dataset comportamento normale
PRIVATE DATA - NOT LOADED.to_csv(PRIVATE DATA - NOT LOADED,index=False)


# ### Creazione dataframe di comportamento anomalo tramite differenza

PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED)



df_old = pd.read_csv(output_file_artf_post_filter)
df_old['DATE_TIME'] = pd.to_datetime(df_old['DATE_TIME'])



df_anom = df_old[~df_old.apply(tuple,1).isin(PRIVATE DATA - NOT LOADED.apply(tuple,1))]



# Salvataggio dataset comportamento anomalo
df_anom.to_csv(output_file_artf_anom_beh,index=False)


# # Normalizzazione dei dati tarati su comportamento normale


from sklearn.preprocessing import MinMaxScaler



output_file_artf_n_norm = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'
output_file_artf_a_norm = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED.csv'



norm_data = pd.read_csv(output_file_artf_normal_beh)
anom_data = pd.read_csv(output_file_artf_anom_beh)

date_norm = norm_data['DATE_TIME']
date_anom = anom_data['DATE_TIME']
norm_data = norm_data.drop(['DATE_TIME'],axis=1)
anom_data = anom_data.drop(['DATE_TIME'],axis=1)



norm_data.head(1)



anom_data.head(1)



# Utilizzare lo stesso scaler per l'intero Dataset!!!
scaler = MinMaxScaler().fit(norm_data.values)
norm_data[norm_data.columns] = scaler.transform(norm_data.values)
anom_data[anom_data.columns] = scaler.transform(anom_data.values)



norm_data['DATE_TIME'] = date_norm
anom_data['DATE_TIME'] = date_anom


norm_data.to_csv(output_file_artf_n_norm,index=False)
anom_data.to_csv(output_file_artf_a_norm,index=False)


