#!/usr/bin/env python
# coding: utf-8

# Per automatizzare il processo, ricerca di una cartella di addestramento rete neurale:




import os
from os import path
import pickle
import json





def recursive_data_net(_path_nets, _list_data_net = []):
    """
    Ricerca le associazioni:
    (cartella addestramento, file modello rete neurale, training salvato rete neurale, file percentile se presente, file iperparametri se presente)
    nel percorso richiesto. 
    Lo restituisce in uscita come lista di tuple.
    
    TODO
    Aggiungere tra le tuplea altre info utili come metrica da utilizzare nel calcolo dei percentili
    
    :param _path_nets: str
        percorso dove ricercare cartelle di reti neurali addestrate
    :param _list_data_net: list
        inizializzazione della variabile accumulatore
        
    :return: list
        variabile accumulatore di ricorsione che conterrà le associazioni
        (cartella addestramento, file modello rete neurale, training salvato rete neurale, file percentile, file iperparametri)
    """
    name_net = ''
    train_file = ''
    perc_file = ''
    hyper_file = ''
    for file in os.listdir(_path_nets):
        file = os.path.join(_path_nets, file)
        if os.path.isdir(file):
            # directory. ricorsione in coda
            recursive_data_net(file, _list_data_net)
        else:
            # file regular
            if file.endswith('.h5'):
                name_net = file
            if 'train_norm_set_' in file:
                # mi basta contenga la stringa lungo path altrimenti cambiare controllo
                train_file = file
            if 'percentiles.csv' in file:
                # se presente file percentili
                perc_file = file
            if 'hyper.' in file:
                # se presente file iperparametri
                hyper_file = file
        
    if (name_net != '') & (train_file!= ''):
        # ho i dati di una intera cartella di una rete neurale addestrata 
        #print('\nCartella ',_path_nets, 'data ', train_file, ' model ', name_net)
        _list_data_net += [(_path_nets, name_net, train_file, perc_file, hyper_file)]
    return _list_data_net    





def recursive_data_report_net(_path_nets, _list_data_net = [], _name_report_to_find = 'pred_90_perc.csv'):
    """
    Crea un array di dizionari composto da:
    {
        'nome_net': file modello rete neurale,
        nome_test_[1..N]: dataframe che contiene le predizioni e le colonne target del test decise dalla rete(default perc.90)
    }
    :param _path_nets: str
        percorso dove ricercare le informazioni
    :param _list_data_net: list
        inizializzazione della variabile accumulatore
    :param _name_report_to_find: str
        nome del report da ricercare. contiene un dataframe con le predizioni della rete e le true labels del test
        
    :return: list
        variabile accumulatore di ricorsione che conterrà le associazioni
    """
    name_net = ''
    for file in os.listdir(_path_nets):
        searched = os.path.join(_path_nets, file)
        if searched.endswith('.h5'):
            # sei nella cartella di una rete neurale
            #print(searched)
            info = {}
            name_net = searched
            info['nome_net'] = name_net
            # avvia ciclo sulle cartelle dei test
            for dir_test in os.listdir(_path_nets):
                path_dir_test = os.path.join(_path_nets, dir_test)
                if os.path.isdir(path_dir_test):
                    name_test = dir_test
                    report_test = os.path.join(path_dir_test, _name_report_to_find)
                    info[name_test] = report_test
            _list_data_net.append(dict(info))
            #print(info)
            #print(_list_data_net)
        else:
            # riavvia ricorsione in coda solo se dir. ricerca cartella di una rete neurale
            if os.path.isdir(searched):
                recursive_data_report_net(searched, _list_data_net)
    
    return _list_data_net    


# Metodi supportati per il calcolo dell'errore di predizione:




def calculate_errors_predict_mse(_model, _data_norm):
    """
    Calcolo errore di predizione con MSE 
    
    :param _model: Model
        modello rete neurale Keras
    :param _data_norm: numpy
        dati normalizzati per calcolo degli errori di predizione
    
    :return: dict
        mse calcolato
    :return: numpy
        array dell'errore di predizione di ogni istanza ricevuta in ingresso 
    :return: numpy
        predizioni della rete
    """
    pred_norm = _model.predict(_data_norm)
    
    n_all_samples, nn_all_columns = _data_norm.shape
    errors_normal = [0] * n_all_samples
    squared_errors_normal = {}

    for j in range(nn_all_columns):
        squared_errors_normal[j] = []
    for i in range(n_all_samples):
        for j in range(nn_all_columns):
            squared_errors_normal[j].append((pred_norm[i][j] - _data_norm[i][j])*(pred_norm[i][j] - _data_norm[i][j]))

    for j in squared_errors_normal.keys():
        for i in range(len(squared_errors_normal[j])):
            if errors_normal[i] < squared_errors_normal[j][i]:
                errors_normal[i] = squared_errors_normal[j][i]
    
    return squared_errors_normal, errors_normal, pred_norm





def calculate_errors_predict_mae(_model, _data_norm):
    """
    Calcolo errore di predizione con MSE 
    
    :param _model: Model
        modello rete neurale Keras
    :param _data_norm: numpy
        dati normalizzati per calcolo degli errori di predizione
    
    :return: dict
        mse calcolato
    :return: numpy
        array dell'errore di predizione di ogni istanza ricevuta in ingresso 
    :return: numpy
        predizioni della rete
    """
    pred_norm = _model.predict(_data_norm)
    
    n_all_samples, nn_all_columns = _data_norm.shape
    errors_normal = [0] * n_all_samples
    abs_errors_normal = {}

    for j in range(nn_all_columns):
        abs_errors_normal[j] = []
    for i in range(n_all_samples):
        for j in range(nn_all_columns):
            abs_errors_normal[j].append(abs(pred_norm[i][j] - _data_norm[i][j]))

    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]
    
    return abs_errors_normal, errors_normal, pred_norm


# Creazione file percentile basato sul principio soglia di errore predizione rete neurale tarata su comportamento desiderato.
# 
# Successivamente l'addestramento di una rete neurale creo il file dei percentili per velocizzare le valutazioni successive delle soli reti non scartate.




import csv
import numpy as np





def calculate_threshold_from_errors(_errors_normal, _percentile = 90):
    """
    Dagli errori di predizioni di ogni istanza ricevuto in ingresso, calcola il percentile scelto e lo restituisce
    
    :param _errors_normal: numpy
        array dell'errore di predizione di ogni istanza
    :param _percentile: int
        l'n scelto per il calcolo del percentile
        
    :return: float
        il valore dell'n-esimo percentile scelto
    """
    error_threshold = np.percentile(np.asarray(_errors_normal), _percentile)
    #print("Percentile: %s (threshold: %s)" % (_percentile, error_threshold))
            
    return error_threshold





def generate_file_percentile(_loaded_model, _data_to_train, _path_net, _percentiles_test = [90,92,95,99], _metric = 'mse'):
    """
    Crea un file contentente il valore del percentile con cui il modello può eseguire una classificazione binaria.
    Calcolando l'errore di predizione, secondo la metrica scelta, dei dati in ingresso
    e in uscita dalla rete viene individuato il valore di soglia per il calcolo dell'n-esimo percentile.
    Se tarato sul comportamento individuabile dalla rete neurale correttamente addestrata individua un valore di soglia
    per discrimare i valori 'simil training' da 'sconosciuti'.
    Supporta diverse metriche
    
    :param _loaded_model: Model
        modello rete neurale Keras
    :param _data_to_train: numpy
        dati normalizzati per calcolo degli errori di predizione
    :param _path_net: str
        percorso in cui salvare il file del percentile
    :param _percentiles_test: list
        valori su cui eseguire il calcolo dell'n-percentile
    :param _metric: str
        metrica da utilizzare. Supporto MAE, MSE(default)

    :return: numpy
        l'errore sui dati secondo la metrica scelta
    """
    
    if _metric == 'mae':
        _, errors_normal, _ = calculate_errors_predict_mae(_loaded_model, _data_to_train)
    else:
        # mse
        _, errors_normal, _ = calculate_errors_predict_mse(_loaded_model, _data_to_train)
    
    
    columns_perc = []
    row_perc = []
    for perc in _percentiles_test:
        error_threshold = calculate_threshold_from_errors(errors_normal, perc)
        columns_perc += [str(perc)]
        row_perc += [error_threshold]
    
    with open(os.path.join(_path_net, 'percentiles.csv'),'w', newline='') as fd:
        wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
        wr.writerow(columns_perc)
        wr.writerow(row_perc)
    
    return errors_normal





def gen_pred_labels(_errors, _error_threshold):
    """
    Dagli errori ottenuti utilizza principio a soglia per determinare predizione di una rete neurale.
    
    :param _errors: numpy
        gli errori di predizione di un rete neurale
    :param _error_threshold: float
        la soglia per la classificazione binaria
    
    :return: list
        le predicted labels della rete neurale
    """

    predictions = []
    for e in _errors:
        if e > _error_threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# Generazione file predizione di una rete:




import tensorflow as tf
from keras import backend as K
import pandas as pd





def multiple_gen_percentile_nets(_path_nets, _percentiles_test = [90,92,95,99], _train_file = ''):
    """
    Per ogni cartella di addestramento di una rete neurale trovata,
    genera i file per il calcolo del percentile
    
    :param _path_nets: str
        percorso dove ricercare cartelle di reti neurali addestrate
    :param _percentiles_test: list
        valori su cui eseguire il calcolo dell'n-percentile
    """
    
    #array (folder, h5, csv)
    data_nets = recursive_data_net(_path_nets, [])
    for able_to_gen in data_nets:
        dir_net = able_to_gen[0]
        name_net = able_to_gen[1]
        if _train_file == '':
            data_file = able_to_gen[2]
            data_reloaded = pd.read_csv(data_file, sep=',',header=None).values
        else:
            data_file = _train_file
            data_reloaded = pd.read_csv(data_file).values
        hyper_file = able_to_gen[4]
        
        if hyper_file.endswith('.json'):
            #json file
            dict_hyper = json.load(open(hyper_file, 'rb'))
            metric = dict_hyper['loss_func']
        else:
            # pickle file
            dict_hyper = pickle.load(open(hyper_file, 'rb'))
            metric = dict_hyper['loss_func']
            
        loaded_model = tf.keras.models.load_model(name_net)
        
        errors = generate_file_percentile(loaded_model, data_reloaded, dir_net, _percentiles_test, metric)      
        print('File percentile creato per rete nella cartella ',dir_net, ' metrica:',metric)


# Generazione e salvataggio della predizione di una rete neurale:




def fast_predict_np_and_save(_name_net, _data_to_predict, _path, _name_file_csv = 'pred_data'):
    """
    Predict della rete neurale dei dati in ingresso.
    Salvataggio del risultato nella directory scelta
    
    :param _name_net: str
        file h5 della rete neurale
    :param _data_to_predict: numpy
        dati normalizzati forniti in ingresso alla rete addestrata
    :param _path: str
        percorso dove salvare il risultato
    :param _name_file_csv: str
        nome del file di output della predizione della rete
    """
    model = tf.keras.models.load_model(_name_net)
    pred_data = model.predict(_data_to_predict)
    np.savetxt(os.path.join(_path, _name_file_csv + '.csv'), pred_data, delimiter=',')
    print('Predizione rapida completata. Salvataggio ',_name_file_csv,'.csv in ', _path)





def fast_predict_df_and_save(_name_net, _df_to_predict, _path, _name_file_csv = 'pred_data', _column_to_exclude = []):
    """
    Predict della rete neurale dei dati in ingresso.
    Salvataggio del risultato nella directory scelta
    
    :param _name_net: str
        file h5 della rete neurale
    :param _df_to_predict: Dataframe
        dataframe da passare in ingresso alla rete
    :param _path: str
        percorso dove salvare il risultato
    :param _name_file_csv: str
        nome del file di output della predizione della rete
    :param _column_to_exclude: list
        la lista delle colonne del Dataframe da escludere dall'input della rete. 
        Recuperate poi per il salvataggio
    """
    
    model = tf.keras.models.load_model(_name_net)
    
    df_temp = _df_to_predict.copy()
    df_temp = df_temp.drop(_column_to_exclude, axis = 1)
    
    pred_data = model.predict(df_temp)
    
    df_pred = pd.DataFrame(pred_data, columns=df_temp.columns)
    for column in _column_to_exclude:
        if column in _df_to_predict.columns:
            df_pred[column] = _df_to_predict[column]
    
    df_pred.to_csv(os.path.join(_path, _name_file_csv + '.csv'),index=False)
    print('Predizione rapida completata. Salvataggio ',_name_file_csv,'.csv in ', _path)


# **Parte Hyperopt:**




from sklearn.metrics import classification_report,accuracy_score,confusion_matrix





def generate_predicted_using_percentile_with_anom_dataset(_model, _train_norm, _data_norm, _data_anom, _metric = 'mse', 
                                                          _percentile = 90):
    """
    DEPRECATED
    Versione facilitata(PRIVATE DATA - NOT LOADED)
    
    Calcolo degli errori di predizione secondo la metrica scelta.
    Generazione della predizione della rete in base all'accuratezza sul testset normale e anomalo.
    
    :param _model: Model
        autoencoder creato. Keras model
    :param _train_norm: numpy
        punti normali del training
    :param _data_norm: numpy
        punti normali di testing
    :param _data_anom: numpy
        punti anomali di testing
    :param _metric: str
        metrica da utilizzare
    :param _percentile: int
        n del percentile

    :return: list
        true labels testset
    :return: list
        predicted labels testset
    :return: numpy
        l'errore del testset secondo la metrica scelta
    :return: float
        la soglia del classificatore binario
    """
    n_samples, n_columns = _data_norm.shape
    a_samples, a_columns = _data_anom.shape
    if _metric == 'mae':
        _, errors_normal_train, _ = calculate_errors_predict_mae(_model, _train_norm)
        _, errors_normal, _ = calculate_errors_predict_mae(_model, _data_norm)
        _, errors_anom, _ = calculate_errors_predict_mae(_model, _data_anom)
    else:
        # mse
        _, errors_normal_train, _ = calculate_errors_predict_mse(_model, _train_norm)
        _, errors_normal, _ = calculate_errors_predict_mse(_model, _data_norm)
        _, errors_anom, _ = calculate_errors_predict_mse(_model, _data_anom)
    
    # Tarata soglia su tutto il training normale
    error_threshold = calculate_threshold_from_errors(errors_normal_train, _percentile)
            
    # Analisi test - parte normale e anomala
    classes_normal = [0] * n_samples
    classes_anom = [1] * a_samples
    errors = errors_normal + errors_anom
    classes = classes_normal + classes_anom

    
    predictions = gen_pred_labels(errors, error_threshold)
    
    return classes, predictions, errors, error_threshold





def calc_errors_with_norm_dataset(_model, _train_norm, _data_norm, _metric = 'mse', 
                                                          _percentile = 90):
    """
    Calcolo degli errori di predizione secondo la metrica scelta.
    Calcolo della soglia per il classificatore binario
    
    :param _model: Model
        autoencoder creato. Keras model
    :param _train_norm: numpy
        punti normali del training
    :param _data_norm: numpy
        punti normali di testing
    :param _metric: str
        metrica da utilizzare
    :param _percentile: int
        n del percentile

    :return: float
        la soglia del classificatore binario
    :return: numpy
        l'errore del testset normale secondo la metrica scelta
    """
    if _metric == 'mae':
        _, errors_normal_train, _ = calculate_errors_predict_mae(_model, _train_norm)
        _, errors_normal, _ = calculate_errors_predict_mae(_model, _data_norm)
    else:
        # mse
        _, errors_normal_train, _ = calculate_errors_predict_mse(_model, _train_norm)
        _, errors_normal, _ = calculate_errors_predict_mse(_model, _data_norm)
    
    # Tarata soglia su tutto il training normale
    error_threshold = calculate_threshold_from_errors(errors_normal_train, _percentile)
    
    return error_threshold, errors_normal





def evaluate_acc_model_with_anom_dataset(_model, _train_norm, _data_norm, _data_anom, _metric = 'mse', _percentile = 90):
    """
    DEPRECATED
    Versione facilitata(PRIVATE DATA - NOT LOADED)
    
    Calcolo accuratezza su testset composto da dati normali e anomali secondo la metrica scelta.
    Modello neurale convertito in classificatore binario tramite principio a soglia.
    
    :param _model: Model
        autoencoder creato. Keras model
    :param _train_norm: numpy
        punti normali del training
    :param _data_norm: numpy
        punti normali di testing
    :param _data_anom: numpy
        punti anomali di testing
    :param _metric: str
        metrica da utilizzare
    :param _percentile: int
        n del percentile

    :return: float
        valore accuratezza classificatore binario
    :return: 
        report completo
    :return: float
        la soglia del classificatore binario
    """
    _data_anom = _data_anom[np.random.choice(_data_anom.shape[0], _data_norm.shape[0], replace=True)]
        
    true_labels, pred_labels, _, error_threshold = generate_predicted_using_percentile_with_anom_dataset(_model, _train_norm, 
                                                                                                         _data_norm, _data_anom, _metric, _percentile)
    
    class_report = classification_report(true_labels, pred_labels)
    #print ("Classification Report: ")
    #print (class_report)
    #print ("")
    
    return accuracy_score(true_labels, pred_labels), class_report, error_threshold





def evaluate_acc_model_with_norm_dataset(_model, _train_norm, _data_norm, _metric = 'mse', _percentile = 90):
    """
    Calcolo accuratezza su testset composto da dati normali secondo la metrica scelta.
    Modello neurale convertito in classificatore binario tramite principio a soglia.
    
    :param _model: Model
        autoencoder creato. Keras model
    :param _train_norm: numpy
        punti normali del training
    :param _data_norm: numpy
        punti normali di testing
    :param _metric: str
        metrica da utilizzare
    :param _percentile: int
        n del percentile

    :return: float
        valore accuratezza classificatore binario
    :return: 
        report completo
    :return: float
        la soglia del classificatore binario
    """
    error_threshold, errors_normal = calc_errors_with_norm_dataset(_model, _train_norm, _data_norm, _metric, _percentile)

    true_labels = [0] * _data_norm.shape[0]

    pred_labels = gen_pred_labels(errors_normal, error_threshold)
    
    class_report = classification_report(true_labels, pred_labels)
    #print ("Classification Report: ")
    #print (class_report)
    #print ("")
    
    return accuracy_score(true_labels, pred_labels), class_report, error_threshold





def evaluate_avg_error_model_semi_supervised(_model, _train_norm, _data_norm, _metric = 'mse', _percentile = 90):
    """
    Metodo per la valutazione della rete basata sull'errore di predizione medio sul testset composto da dati normali.
    Modello neurale convertito in classificatore binario tramite principio a soglia.
    
    :param _model: Model
        autoencoder creato. Keras model
    :param _train_norm: numpy
        punti normali del training
    :param _data_norm: numpy
        punti normali di testing
    :param _metric: str
        metrica da utilizzare
    :param _percentile: int
        n del percentile

    :return: float
        la soglia del classificatore binario
    :return: float
        l'errore medio sul testset
    """
    error_threshold, errors_normal = calc_errors_with_norm_dataset(_model, _train_norm, _data_norm, _metric, _percentile)
  
    avg_error = 0
    for e in errors_normal:
        avg_error += e
    avg_error /= _data_norm.shape[0]
    
    #print('Thr error test:{}'.format(str(error_threshold)))
    #print('Avg_error:{}'.format(str(avg_error)))
    return error_threshold, avg_error


# Generazione plot errori di predizione rete neurale:




import matplotlib.pyplot as plt





def plot_errors_prediction_with_date(_name_net, _errors, _dates, _labels = '',
                                     _percentile = 95, _thr_value = 0, 
                                     _save_plot = False, _path_dir_net = '', _name_test = 'test'):
    """
    Generazione plot errore di predizione rete neurale.
    
    :param _name_net: str
        nome rete neurale
    :param _errors: numpy
        errori di predizione rete neurale
    :param _dates: DatetimeIndex
        asse x del grafico
    :param _labels: list
        true labels, per assegnamento colore. Rosso punti anomali, verde punti normali.
        Se non passato come argomento i punti saranno tutti di colore grigio.
    :param _percentile: int
        n del percentile scelto
    :param _thr_value: float
        valore della soglia per classificatore binario. 
        Oltre soglia il punto è predetto dalla rete come anomalia, sotto soglia come stato normale
    :param _save_plot: bool
        True se si desidera salvare il grafico. False altrimenti
    :param _path_dir_net: str
        percorso directory per salvataggio grafico se abilitato
    :param _name_test: str
        nome del test. Se il salvataggio del grafico è abilitato sarà creata directory con tale nome
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    
    if type(_labels) is pd.core.series.Series: 
        # scatter plot dei punti. punti verdi stati normali, punti rossi stati anomali (true labels)
        colors = {0:'green', 1:'red'}
        ax.scatter(_dates,_errors,s=2, c=_labels.apply(lambda x: colors[x]))
    else:
        ax.scatter(_dates,_errors,s=2,c='grey')
    
    # retta settata al valore della soglia. Oltre la soglia è stato anomalo, sotto soglia è stato normale (predicted labels)
    ax.plot([_dates[0],_dates[-1]], [_thr_value[0],_thr_value[0]], c='yellow', label='Valore soglia')
    
    plt.legend(loc='upper left')
    plt.title('{}, errore di predizione. percentile={}'.format(_name_net,str(_percentile)))
    plt.xlabel('Data')
    plt.grid()
    plt.ylabel('Valore errore di predizione')    
    
    if _save_plot:
        dir_test_normal = os.path.join(_path_dir_net,_name_test)
        if not os.path.exists(dir_test_normal):
            print("Creazione directory {}".format(dir_test_normal))
            os.mkdir(dir_test_normal)
        
        plt.savefig(os.path.join(dir_test_normal, 'test_normal_perc_{}'.format(str(_percentile))));
        print('Plot {} percentile={} done!'.format(_name_net,str(_percentile)))
    else:
        plt.show()





def plot_errors_prediction_with_index(_name_net, _errors, _shape_max, _labels = '',
                                     _percentile = 95, _thr_value = 0, 
                                     _save_plot = False, _path_dir_net = '', _name_test = 'test'):
    """
    Generazione plot errore di predizione rete neurale.
    
    :param _name_net: str
        nome rete neurale
    :param _errors: numpy
        errori di predizione rete neurale
    :param _shape_max: int
        numero max di elementi da plottare lungo l'asse x. Può corrisponde all'indice del test
    :param _labels: list
        true labels, per assegnamento colore. Rosso punti anomali, verde punti normali.
        Se non passato come argomento i punti saranno tutti di colore grigio.
    :param _percentile: int
        n del percentile scelto
    :param _thr_value: float
        valore della soglia per classificatore binario. 
        Oltre soglia il punto è predetto dalla rete come anomalia, sotto soglia come stato normale
    :param _save_plot: bool
        True se si desidera salvare il grafico. False altrimenti
    :param _path_dir_net: str
        percorso directory per salvataggio grafico se abilitato
    :param _name_test: str
        nome del test. Se il salvataggio del grafico è abilitato sarà creata directory con tale nome
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    
    if type(_labels) is pd.core.series.Series:   
        # scatter plot dei punti. punti verdi stati normali, punti rossi stati anomali (true labels)
        colors = {0:'green', 1:'red'}
        ax.scatter(np.arange(_shape_max),_errors,s=2, c=_labels.apply(lambda x: colors[x]))
    else:
        ax.scatter(np.arange(_shape_max),_errors,s=2, c='grey')
    
    # retta settata al valore della soglia. Oltre la soglia è stato anomalo, sotto soglia è stato normale (predicted labels)
    ax.plot([0,_shape_max], [_thr_value[0],_thr_value[0]], c='yellow', label='Valore soglia')
    
    plt.legend(loc='upper left')
    plt.title('{}, errore di predizione. percentile={}'.format(_name_net,str(_percentile)))
    plt.xlabel('Indice')
    plt.grid()
    plt.ylabel('Valore errore di predizione')    
    
    if _save_plot:
        dir_test_normal = os.path.join(_path_dir_net,_name_test)
        if not os.path.exists(dir_test_normal):
            print("Creazione directory {}".format(dir_test_normal))
            os.mkdir(dir_test_normal)
        
        plt.savefig(os.path.join(dir_test_normal, 'test_normal_perc_{}'.format(str(_percentile))));
        print('Plot {} percentile={} done!'.format(_name_net,str(_percentile)))
    else:
        plt.show()


# Per ogni test generare plot percentile, Dataframe per analisi e report riassuntivo




def gen_complete_analysis_net(_name_net, _df_data, _path_net, _df_percentile, _metric = 'mse', 
                              _index_data = '', _labels = '', _name_test = 'test'):
    """
    Generazione report di analisi.
    Per test da analizzare, carico la rete neurale, i suoi valori di soglia e genero i plot degli errori di predizioni.
    All'interna della cartella creata saranno presenti un plot e un Dataframe per ogni percentile scelto.
    Presente un ulteriore report globale con le informazioni della rete neurale, le predizioni e la loro relativa occorrenza.
        
    :param _name_net: str
        nome della rete da analizzare
    :param _df_data: Dataframe
        dati da analizzare con la rete neurale
    :param _path_net: str
        percorso cartella per salvataggio. Preferibile all'interno della stessa cartella della rete neurale 
    :param _df_percentile: Dataframe
        contiene i valori di soglia da utilizzare per la generazione dei plot e delle predizioni della rete neurale
    :param _metric: str
        metrica da utilizzare per generazione predicted labels rete neurale
    :param _index_data: 
        x da utilizzare per i plot. 
        Se campo vuoto utilizza plot ad indici. Altrimenti plot a date
    :param _labels:
        true labels se fornite abilitano colore verde/rosso nel plot per distinzione true labels nei grafici.
        Colore grigio assegnato ai punti se campo vuoto
    :param _name_test: str
        nome da assegnare alla directory per l'analisi
    """
    
    loaded_model = tf.keras.models.load_model(_name_net) # caricamento rete neurale
    
    columns_report = ['Name_net']
    row_report = [_name_net]
    
    print('Rete ', _name_net, ' utilizza metrica:', _metric)
    if _metric == 'mae':
        _, errors, pred_data = calculate_errors_predict_mae(loaded_model, _df_data.values)
    else:
        _, errors, pred_data = calculate_errors_predict_mse(loaded_model, _df_data.values)
    
    for perc in _df_percentile.columns:
        
        pred_labels = gen_pred_labels(errors, _df_percentile[perc][0])
        # un Dataframe per percentile con le predizioni della rete in base alla soglia
        df_pred = pd.DataFrame(pred_data,columns=_df_data.columns)
        
        if type(_index_data) is list: 
            plot_errors_prediction_with_date(_name_net, errors, _index_data, _labels,
                                     _percentile = int(perc), _thr_value = _df_percentile[perc], 
                                     _save_plot = True, _path_dir_net = _path_net, _name_test = _name_test)
            df_pred['DateTime'] = _index_data
        else:
            plot_errors_prediction_with_index(_name_net, errors, _df_data.shape[0], _labels,
                                     _percentile =int(perc), _thr_value = _df_percentile[perc], 
                                     _save_plot = True, _path_dir_net = _path_net, _name_test = _name_test)
        
        if type(_labels) is pd.core.series.Series:    
            df_pred['True'] = _labels
            
            true_labels = _labels.values
            class_report = classification_report(true_labels, pred_labels)

            file_report = os.path.join(_path_net, _name_test +'\\class_report_{}.txt'.format(int(perc)))
            with open(file_report,'w') as fh:
                fh.write(class_report)
        
        df_pred['Pred'] = pred_labels
        df_pred.to_csv(os.path.join(_path_net, _name_test + '\\pred_{}_perc.csv'.format(int(perc))),index=False)
        
        unique, counts = np.unique(pred_labels, return_counts=True)
        
        columns_report += [
                        'Perc_{}'.format(perc),
                        'Occ_{}'.format(perc),
                        'Pred_Perc_{}'.format(perc)]
        row_report += [
                        _df_percentile[perc][0],
                        [dict(zip(unique, counts))],
                        [pred_labels]]
    
    # report completo
    file_report_test = os.path.join(_path_net,_name_test +'\\report.csv')
    
    # sovrascrittura file perchè unico test per cartella
    with open(file_report_test,'w', newline='') as fd:
        wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
        wr.writerow(columns_report)
        wr.writerow(row_report)
    
    print('Net {} done!'.format(_name_net))





def gen_complete_report_test_net(_dir_net, _file_report_metrics, _file_report_predictions):
    """
    Generazione dei due report di un dispositivo.
    Il primo report '{}_metrics' contiene i risultati delle metriche di ogni net.
    Le metriche scelte sono rispettivamente:
        - TN,FP,FN,TP(confusion_matrix)
        - TPR, FNR, FPR, TNR, PPV, FDR, FOR, NPV, Prevalence, ACC, LR+, LR-, DOR, FSCORE, FSCORE_0, _FSCORE1
    I nomi fanno riferimento a (https://en.wikipedia.org/wiki/Precision_and_recall)
    Il secondo report '{}_predictions' contiene, per ogni test applicato ad una net, le true labels, le predicted labels
    e la dimensionalità del set
    
    :param _dir_net: str
        cartella che contiene le reti su cui eseguire il report
    :param _file_report_metrics: str
        percorso e nome del file che contiene le metriche. Spesso _dir_net/{nome_dispositivo}_metrics.csv
    :param _file_report_predictions: str
        percorso e nome del file che contiene le predizioni. Spesso _dir_net/{nome_dispositivo}_predictions.csv 
    """
    info_test = recursive_data_report_net(_dir_net, [])
    columns_report_metrics = ['Name_net']
    columns_report_pred_true = ['Name_net']
    for info_net in info_test:
        # salvo nome net
        row_report_metrics = [info_net['nome_net']]
        row_report_pred_true = [info_net['nome_net']]
        for key, value in info_net.items():
            if(key != 'nome_net'):
                # allora è uno dei test. recupero true, predicted e calcolo valori per report
                column_test = key + '_true_labels'
                if key not in columns_report_pred_true:
                    # solo prima volta per questo test. completo le sue colonne
                    columns_report_pred_true += [column_test, (key + '_pred_labels'), (key + '_dimension')]
                    columns_report_metrics += [(key + '_tn'), (key + '_fp'), (key + '_fn'), (key + '_tp'),
                                               (key + '_TPR'), (key + '_FNR'), (key + '_FPR'), (key + '_TNR'),
                                               (key + '_PPV'), (key + '_FDR'), (key + '_FOR'), (key + '_NPV'),
                                               (key + '_Prevalence'), (key + '_ACC'), (key + '_LR+'), (key + '_LR-'),
                                               (key + '_DOR'), (key + '_FSCORE'), (key + '_FSCORE_0'), (key + '_FSCORE1')]
                df_to_load = pd.read_csv(value)
                true_labels = df_to_load['True']
                pred_labels = df_to_load['Pred']
                #print(key)
                #print(confusion_matrix(true_labels, pred_labels, labels=[0,1]).ravel())
                row_report_pred_true += [[true_labels.tolist()], [pred_labels.tolist()], true_labels.shape[0]]
                tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0,1]).ravel()
                cond_pos = tp + fn
                cond_neg = fp + tn
                total_pop = cond_pos + cond_neg
                pred_con_pos = tp + fp
                pred_con_neg = fn + tn
                tpr = tp / cond_pos
                fnr = fn / cond_pos
                fpr = fp / cond_neg
                tnr = tn / cond_neg
                ppv = tp / pred_con_pos
                rep = classification_report(true_labels, pred_labels, labels=[0,1], output_dict=True)
                row_report_metrics += [ tn,fp,fn,tp,
                                        tpr,fnr,fpr,tnr,
                                        ppv,(fp / pred_con_pos),(fn / pred_con_neg),(tn / pred_con_neg),
                                        (cond_pos / total_pop),((tp + tn) / total_pop),(tpr / fpr),(fnr / tnr),
                                        (tpr / fpr)/(fnr / tnr), rep['weighted avg']['f1-score'], rep['0']['f1-score'],rep['1']['f1-score']]
                #print(row_report_metrics)
        if not path.exists(_file_report_predictions) and not path.exists(_file_report_metrics):
            with open(_file_report_predictions,'w', newline='') as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(columns_report_pred_true)
                wr.writerow(row_report_pred_true)
            with open(_file_report_metrics,'w', newline='') as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(columns_report_metrics)
                wr.writerow(row_report_metrics)
        else:
            with open(_file_report_predictions,'a',newline='') as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(row_report_pred_true)
            with open(_file_report_metrics,'a',newline='') as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(row_report_metrics)





def convert_arr_str_to_numpy(_column_arr_str):
    """
    Prende in ingresso un array di interi salvato come stringa e restituisce il numpy array.
    Visto che alcuni metodi salvano all'interno di un Dataframe come colonna una lista di interi(es. true/predicted labels)
    con questo metodo si può riottenere l'array
    
    :param _column_arr_str: str
        array di interi come stringa (es. '[0, 0, 1, 1, 1, 0]')
    :return: numpy
        array di interi
    """
    return np.array(list(_column_arr_str.replace('[','').replace(']','').replace(',','').replace(' ','')),dtype=int)

