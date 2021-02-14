#!/usr/bin/env python
# coding: utf-8

# # File creazione rete neurale


# Librerie
import pandas as pd


# Import libreria reti neurali:



import lib_nn_generation
import code_test.lib_nn_prediction_analysis as lib_nn_prediction_analysis



config_net = lib_nn_generation.default_autoencoder_hyperopt.copy()

print('Parametri rete:')
config_net


# #### Caricamento dataframe 




dir_artf = 'PRIVATE DATA - NOT LOADED\\preprocess\\'
dir_output_net = 'PRIVATE DATA - NOT LOADED\\'

file_training = dir_artf + 'PRIVATE DATA - NOT LOADED.csv'
file_anom = dir_artf + 'PRIVATE DATA - NOT LOADED.csv' # prima versione basata su accuratezza. Vedere report e/o documentazione

df_norm = pd.read_csv(file_training)

df_anom = pd.read_csv(file_anom) # dataset anomalo normalizzato. Sample randomico eseguito successivamente

directory_name = dir_output_net + 'nets\\hyperopt'


# La rete neurale non prende in ingresso la data 




# Eliminazione date
df_norm = df_norm.drop(['DATE_TIME'],axis=1)
df_anom = df_anom.drop(['DATE_TIME'],axis=1)

data_anom = df_anom.values


# Diversi tentativi:
#     - con tutti gradienti
#     - senza alcun gradiente
#     - senza gradienti delle potenze
#     - senza temperatura: abbandonato perchè per anomalie come derating la temperatura è essenziale
#     
# Esecuzione parti interessate con aggiunta nel nome della directory di riferimento le parti mancanti

# Esempio dati rete:




print('Dimensionalità dati ehp:',df_norm.shape)


# **Eliminazione gradienti potenze**
# 
# Dataset anomalo da cambiare se utilizzato




# **Eliminazione gradienti**
# 
# Dataset anomalo da cambiare se utilizzato


# **Eliminazione stato di macchina sottocarico Se utilizzato necessario sia in training sia in test**




# #### Caricamento datasets esistenti 




dir_net_to_search = 'PRIVATE DATA - NOT LOADED\\nets'
possibile_nets = lib_nn_prediction_analysis.recursive_data_net(dir_net_to_search)
#print('Possibili scelte:\n',possibile_nets)





path_net = possibile_nets[0][0]
name_net = possibile_nets[0][1]
train_file = possibile_nets[0][2]
train_reloaded = pd.read_csv(train_file, sep=',',header=None).values


# **Generazione rete no hyperopt - split abilitato**




lib_nn_generation.create_nets(False, config_net, _directory_save = directory_name, _save_temp_net = True, _max_gen = 1,
                      _df_norm = df_norm)


# **Generazione rete hyperopt**




lib_nn_generation.run_hyperopt_semisupervised(False, 'avg_net', config_net, _directory_save = directory_name, _save_temp_net = True,
                      _df_norm = df_norm, _max_eval_hopt = 1)


# **Generazione rete hyperopt - con guida hyperopt accuratezza dataset anomalo**
# 
# Approccio abbandonato




lib_nn_generation.run_hyperopt_acc_with_anom(False, config_net, _directory_save = directory_name, _save_temp_net = True,
                      _df_norm = df_norm, _data_anom = data_anom,_max_eval_hopt = 1)


# **Fast predizione rete - train**




dir_net_to_search = '..\\..\\PRIVATE DATA - NOT LOADED\nets'
possibile_nets = lib_nn_prediction_analysis.recursive_data_net(dir_net_to_search)
#print('Possibili scelte:\n',possibile_nets)





path_net = possibile_nets[0][0]
name_net = possibile_nets[0][1]
train_file = possibile_nets[0][2]

train_reloaded = pd.read_csv(train_file, sep=',',header=None).values

lib_nn_prediction_analysis.fast_predict_np_and_save(name_net, train_reloaded, path_net, 'train_pred')


# #### Creazione percentili




dir_nets = '..\\..\\dataset\\PRIVATE DATA - NOT LOADED\\nets'
percentiles_test = [90, 92, 95, 99]





lib_nn_prediction_analysis.multiple_gen_percentile_nets(dir_nets, _metric='mse')







