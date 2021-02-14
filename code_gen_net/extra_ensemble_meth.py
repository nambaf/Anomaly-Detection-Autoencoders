#!/usr/bin/env python
# coding: utf-8

# Import libreria reti neurali:


import tensorflow as tf
from keras import backend as K
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import lib_nn_generation
import code_test.lib_nn_prediction_analysis as lib_nn_prediction_analysis



dir_net_1 = 'PRIVATE DATA - NOT LOADED\\n_drop_relu_mse'
dir_net_2 = 'PRIVATE DATA - NOT LOADED\\y_drop_relu_mse'
info_nets_1 = lib_nn_prediction_analysis.recursive_data_net(dir_net_1, [])
info_nets_2 = lib_nn_prediction_analysis.recursive_data_net(dir_net_2, [])


# Recupero le reti neurali su cui desidero effettuare combinare le predizioni:


file_chp_net_1 = info_nets_1[0][1]
file_chp_net_2 = info_nets_1[1][1]
file_chp_net_3 = info_nets_2[0][1]



def generate_extra_info(_file_hyper, _file_perc):
    """
    Recupera la metrica per il calcolo dell'errore di ricostruzione e un valore di soglia dai file di una rete neurale.
    
    :param _file_hyper: path
        percorso del file degli iperparametri di una rete neurale da cui ricavare la funzione di perdita utilizzata
    :param _file_perc: path
        percorso del file dei valori di soglia di una rete neurale da cui ricavare un threshold
    
    :return: tuple
        tupla con metrica e valore di soglia
    """
    if _file_hyper.endswith('.json'):
        #json file
        dict_hyper = json.load(open(_file_hyper, 'rb'))
        metric = dict_hyper['loss_func']
    else:
        # pickle file
        dict_hyper = pickle.load(open(_file_hyper, 'rb'))
        metric = dict_hyper['loss_func']
    
    df_perc = pd.read_csv(_file_perc)
    perc = df_perc['92'][0]
    
    return (metric, perc)




# preparazione modelli e file necessari
models = list()
extra_infos = list() # un array che contiene file hyper e perc associati
model_1 = tf.keras.models.load_model(file_chp_net_1)
models.append(model_1)
extra_infos.append(generate_extra_info(info_nets_1[0][4], info_nets_1[0][3]))
model_2 = tf.keras.models.load_model(file_chp_net_2)
models.append(model_2)
extra_infos.append(generate_extra_info(info_nets_1[1][4], info_nets_1[1][3]))
model_3 = tf.keras.models.load_model(file_chp_net_3)
models.append(model_3)
extra_infos.append(generate_extra_info(info_nets_2[0][4], info_nets_2[0][3]))



# Con una rete neurale che restituisce già la classe target il passaggio per la definizione della classe label non è necessario. Tuttavia le nostre reti sono autoencoder la cui uscita è la ricostruzione dell'input.
# Bisogna quindi 'caricare' percentili ed errori di ricostruzioni(il nostro discriminante) per ottenere da una ricostruzione di uno stato la sua classe predetta.



def easy_load_prediction(_loaded_model, _df_to_predict, _metric, _threshold):
    """
    Metodo per ottenere le predicted labels(anomalo 1, normale 0) di un autoencoder caricato.
    Per una corretta esecuzione utilizzare come metrica e valore di soglia quelli ritrovabili nella cartella di addestramento.
    
    :param _loaded_model: Model
        modello neurale caricato
    :param _df_to_predict: Dataframe
        il Dataframe da analizzare
    :param _metric: str
        metrica per il calcolo dell'errore di ricostruzione. 'mse' a default, 'mae' supportato
    :param _threshold: double
        valore di soglia per classificatore binario su errore di ricostruzione della rete neurale
    
    :return: numpy
        le predicted labels della rete neurale su _df_to_predict analizzato
    """
    if _metric == 'mae':
        _, errors, pred_data = lib_nn_prediction_analysis.calculate_errors_predict_mae(_loaded_model, _df_to_predict.values)
    else:
        _, errors, pred_data = lib_nn_prediction_analysis.calculate_errors_predict_mse(_loaded_model, _df_to_predict.values)
    
    pred_labels = lib_nn_prediction_analysis.gen_pred_labels(errors, _threshold)
    return pred_labels


# Si definisce una funzione per la predizione di un elenco di reti neurali. Ogni rete prevede una certa label. Se per la maggioranza è classificata come anomala allora il risultato finale sarà classe anomala. Idem per label normale.



def ensemble_predictions(_models, _df_to_predict, _extra_info_models):
    """
    Avvia l'ensemble prediction.
    :param _models: list
        la lista dei modelli neurali caricati
    :param _df_to_predict: Dataframe
        il Dataframe da analizzare
    :param _extra_info_models: list
        lista di tuple con i file che contengono la metrica da utilizzare e la soglia del percentile per ogni rete
    """
    predictions = []
    for index, model in enumerate(_models):
        
        pred_labels = easy_load_prediction(model, _df_to_predict, _extra_info_models[index][0], _extra_info_models[index][1])
        predictions.append(pred_labels)
    
    # funzione predizione. esistono diverse metodologie. un esempio è sistema di voting a maggioranza.
    result = np.sum(predictions, axis = 0)
    # se per la maggioranza è anomalo allora la label finale è 1, altrimento è 0
    result[result < round(len(models)/2)] = 0
    result[result >= round(len(models)/2)] = 1
    return result


# Caricamento del subset da analizzare. Il passaggio relativo al caricamento delle reti e delle informazioni necessarie per la predizione si consiglia di memorizzarle così da velocizzare il processo

subset = 'PRIVATE DATA - NOT LOADEDv'
# Carico il Dataframe ma tengo colonna Anom per classification_report finale.
df_to_predict = pd.read_csv(subset)
if 'Anom' in df_to_predict.columns:
    labels = df_to_predict['Anom']
    df_to_predict = df_to_predict.drop(columns=['Anom']) # da rimuovere per predict


# Qua avviene l'utilizzo dell'ensemble
predictions = ensemble_predictions(models, df_to_predict, extra_infos)




unique, counts = np.unique(predictions, return_counts=True)
print('La dimensionalità del set è:',df_to_predict.shape[0])
print('La rete ensemble ha predetto:',dict(zip(unique, counts)))



class_report = classification_report(labels, predictions)
print ("Classification Report: ")
print (class_report)


