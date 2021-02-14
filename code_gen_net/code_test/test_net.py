#!/usr/bin/env python
# coding: utf-8

# # Studio testset: rete neurale


dir_chp = 'PRIVATE DATA - NOT LOADED'

# lista dove ricercare le reti neurali
dir_nets = dir_chp + '\\PRIVATE DATA - NOT LOADED'
#dir_nets = dir_chp + '\\PRIVATE DATA - NOT LOADED'





# lista dei test da analizzare con tutte le reti
tests = [
    'PRIVATE DATA - NOT LOADED'
]
# Assumo siano nella stessa directory di riferimento
dir_tests = dir_chp





# grafico analisi train
tests = [
    'PRIVATE DATA - NOT LOADED'
]
# Assumo siano nella stessa directory di riferimento
dir_tests = dir_chp


# Librerie:

from lib_nn_prediction_analysis import recursive_data_net, calculate_errors_predict_mse, calculate_errors_predict_mae, plot_errors_prediction_with_date, plot_errors_prediction_with_index, gen_complete_analysis_net, gen_complete_report_test_net
    
import pickle
import json
import pandas as pd


# **Generazione analisi completa**
# 
# Per ogni test scelto, per ogni rete neurale scelta genera un report completo composto da plot sugli errori di predizioni, Dataframe con le predizioni della rete e le relative labels. Compreso un report riassuntivo.
# 
# Salvataggio di una directory contenente tutti i file dell'analisi nello stesso percorso della rete neurale.
# 
# Può richiedere diverso tempo se caricate molte reti e molti test. 
# Tuttavia, la creazione incrementale permette un controllo in caso di fallimenti o stop




net_to_plot = recursive_data_net(dir_nets, [])

for tuples in net_to_plot:
    path_net = tuples[0]
    name_net = tuples[1]
    train_file = tuples[2] # per controllo shape
    perc_net_file = tuples[3] # assumo sia presente se voglio plot, altrimenti errore al caricamento
    hyper_file = tuples[4] # assumo sia presente. Per caricamento metrica corretta
    
    df_perc = pd.read_csv(perc_net_file)
    df_train = pd.read_csv(train_file, sep=',',header=None)
    
    if hyper_file.endswith('.json'):
        #json file
        dict_hyper = json.load(open(hyper_file, 'rb'))
        metric = dict_hyper['loss_func']
    else:
        # pickle file
        dict_hyper = pickle.load(open(hyper_file, 'rb'))
        metric = dict_hyper['loss_func']
    
    for test_to_run in tests:
        # nome del test da analizzare
        df_test = pd.read_csv(dir_tests + '\\'+test_to_run+'.csv') # Dataframe normalizzato del test da analizzare
        
        if(df_train.shape[1] < 10):
            columns_to_drop = ['PRIVATE DATA - NOT LOADED']
            df_test = df_test.drop(columns=columns_to_drop) # no grad net
        index_data = '' # '' se x=indici, dates altrimenti
        if 'DateTime' in df_test.columns:
            df_test['DateTime'] = pd.to_datetime(df_test['DateTime'],yearfirst=True)
            index_data = list(df_test['DateTime'])
            df_test = df_test.drop(columns=['DateTime']) # da rimuovere per predict
            
        labels = '' # '' se non presenti true labels, list altrimenti
        if 'Anom' in df_test.columns:
            labels = df_test['Anom']
            df_test = df_test.drop(columns=['Anom']) # da rimuovere per predict
        
        gen_complete_analysis_net(name_net, df_test, path_net, df_perc, _metric = metric, 
                              _index_data = index_data, _labels = labels, _name_test = test_to_run)


# **Creazione report**


dir_blind = '..\\..\\..\\dataset\\PRIVATE DATA - NOT LOADED'
dir_net = dir_blind + '\\PRIVATE DATA - NOT LOADED'
file_report_metrics = dir_net + '\\PRIVATE DATA - NOT LOADED_metrics.csv'
file_report_predictions = dir_net + '\\PRIVATE DATA - NOT LOADED_predictions.csv'



gen_complete_report_test_net(dir_net,file_report_metrics,file_report_predictions)




dir_net = dir_blind + '\\transfer_learning'
file_report_metrics = dir_net + '\\PRIVATE DATA - NOT LOADED_transf_metrics.csv'
file_report_predictions = dir_net + '\\PRIVATE DATA - NOT LOADED_transf_predictions.csv'




gen_complete_report_test_net(dir_net,file_report_metrics,file_report_predictions)













