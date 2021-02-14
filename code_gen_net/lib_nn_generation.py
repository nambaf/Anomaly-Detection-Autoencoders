#!/usr/bin/env python
# coding: utf-8

# Split dei dati:




from sklearn.model_selection import train_test_split
import numpy as np





def split_train_test(_data_norm, _test_size = 0.2):
    """
    Esegue lo split in due parti di un Dataframe.
    
    :param _data_norm : Dataframe
        punti normali
    :param _test_size : float
        percentuale punti di test
    
    :return: numpy
        training set
    :return: numpy
        test set
    """
    # Split
    x_train_norm, x_test_norm = train_test_split(_data_norm.values, test_size = _test_size)
    
    return x_train_norm, x_test_norm





def split_train_test_valid(_data_norm, _test_size = 0.3, _valid_size = 0.4):
    """
    Esegue lo split in tre parti di un Dataframe.
    
    :param _data_norm : Dataframe
        punti normali
    :param _test_size : float
        percentuale punti di test
    :param _valid_size : float
        percentuale punti di validation
    
    :return: numpy
        training set
    :return: numpy
        test set
    :return: numpy
        validation set
    :return: numpy
        test+validation set
    """   
    # Split
    x_train_norm, x_test = train_test_split(_data_norm.values, test_size = _test_size)
    x_test_norm, x_valid_norm = train_test_split(x_test, test_size = _valid_size)
    
    return x_train_norm, x_test_norm, x_valid_norm, x_test


# Parametri default rete neurale autoencoder




default_autoencoder = {
    'epochs': 50, # totale epoche addestramento
    'batch_size': 64, # dimensione batch
    'shuffle': True, # abilitazione shuffle .fit
    # iperparametri
    'num_layers': 2,
    'num_unit_1': 20, # numero neuroni di un layer
    'num_unit_2': 20,
    'actv_func': 'relu', # funzione di attivazione layer
    'actv_func_out': 'relu', # funzione di attivazione uscita
    'loss_func': 'mse', # funzione di perdita
    'l1_reg': 0.00001, # regolarizzatore l1 layer
    'learn_rate_opt': 0.0001, # learning rate
    'optimizer': 'adam', # ottimizzatore
    'drop_enabled': False, # True se abilitati layer di Dropout, False altrimenti
    'drop_factor':0.1 # se True drop_enabled, fattore di Dropout.
}


# Struttura autoencoder parametrica:




import tensorflow as tf
from keras import backend as K 
from tensorflow.python.keras.models import load_model, Sequential, Model
from tensorflow.python.keras import optimizers, initializers, regularizers
from tensorflow.python.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input
from tensorflow.python.keras.layers import UpSampling1D, Dropout, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.utils import plot_model, np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,TensorBoard
from tensorflow.python.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.losses import mse, binary_crossentropy
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier





def autoencoder(_n_features, _hparams = default_autoencoder):
    """
    Creazione struttura autoencoder Keras parametrica.
    
    :param _n_features: int
        numero input rete neurale. Configurazione tensore principale
    :param _hparams: dict
        parametri per la configurazione della rete
    
    :return: Model
        autoencoder creato. Keras model
    """
    input_layer = Input(shape=(_n_features,))
    
    for layer in range(1,int(_hparams['num_layers'])+1):
        hidden = Dense(units=int(_hparams['num_unit_'+str(layer)]), activation=_hparams['actv_func'], activity_regularizer=regularizers.l1(_hparams['l1_reg']))(input_layer if layer == 1 else hidden)
        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)
    
    for layer in reversed(range(1,int(_hparams['num_layers'])+1)):
        hidden = Dense(units=int(_hparams['num_unit_'+str(layer)]), activation=_hparams['actv_func'], activity_regularizer=regularizers.l1(_hparams['l1_reg']))(hidden)
        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)
        
    output_layer = Dense(_n_features, activation=_hparams['actv_func_out'])(hidden)
    
    autoencoder = Model(input_layer, output_layer)
            
    autoencoder.compile(optimizer=_hparams['optimizer'], loss=_hparams['loss_func'])
    
    return autoencoder


# Generazione plot history dell'addestramento delle reti




import matplotlib.pyplot as plt





def train_loss_plot(_history, _save = False, _name = "plot_training_valid_loss"):
    """
    Plot history addestramento rete neurale
    :param _history: History
        dizionario con le informazioni dell'addestramento
    :param _save: bool
        True se si desidera salvare il file png del plot
    :param _name = str
        nome del file salvato. Se abilitato
    """
    plt.plot(_history.history['loss'])
    plt.plot(_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    if _save:
        plt.savefig(_name);
    
    plt.show()


# Generazione multipla di reti neurali con salvataggio informazioni per la ricostruzione e analisi




import os
from os import path
import time
import json
import csv





def create_nets(_load_dataset, _param_net = default_autoencoder, _directory_save = 'default_nets', _save_temp_net = True, _max_gen = 100,
                      _df_norm = '', x_train_norm = '', x_valid_norm = ''):
    """
    Generazione rete neurale
    
    :param _load_dataset : bool
        True se si desidera utilizzare una partizione di train/test/validation già esistente. 
        False se si possiede il solo Dataframe di punti normali.
        Entrambe le forme utilizzano dati già normalizzati
    :param _param_net : dict
        parametri di configurazione rete. Per creazione, addestramento... 
    :param _directory_save : str
        percorso della cartella di destinazione per il salvataggio delle reti
    :param _save_temp_net : bool
        True se si desidera salvare tutta la cartella con i relativi files per il recupero della rete neurale generata.
        False altrimenti.
    :param _max_gen : int
        numero massimo di reti neurali da generare
    :param _df_norm : Dataframe
        punti normali già normalizzati su cui eseguire split in train e test/validation. Utilizzato se _load_dataset è FALSE
    :param x_train_norm : numpy array
        punti normali già normalizzati e pre caricati di addestramento. Utilizzato se _load_dataset è TRUE
    :param x_valid_norm : numpy array
        punti normali già normalizzati e pre caricati di validazione. Utilizzato se _load_dataset è TRUE
    """
    
    gen_net = 0
    
    while(gen_net < _max_gen):
    
        if not _load_dataset:
            # necessito split
            print('Eseguo split punti normali, creazione train e validation sets')
            x_train_norm, x_valid_norm = split_train_test(_df_norm)
        else:
            print('Saranno utilizzati dataset caricati')
            # utilizzo parametri metodo
        
        n_features = x_train_norm.shape[1]
        
        # Creo nuova struttura autoencoder
        autoencoder_model = autoencoder(n_features, _param_net)
        
        startime_run = str(time.time())
        if not os.path.exists(_directory_save):
            print("Creazione directory {}".format(_directory_save))
            os.mkdir(_directory_save)
        
        # Aggiunta callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=1e-5) 
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)
        
        history = autoencoder_model.fit(x_train_norm, x_train_norm, epochs=_param_net['epochs'], 
                validation_data=(x_valid_norm, x_valid_norm),
                shuffle=_param_net['shuffle'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0)
        
        if _save_temp_net:
            directory_run = os.path.join(_directory_save, startime_run)
            # Creazione cartella della specifica rete neurale generata
            os.mkdir(directory_run)
            
            # Salvataggio struttura
            file_summary = os.path.join(directory_run, 'summary.txt')
            with open(file_summary,'w') as fh:
                autoencoder_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            # Salvataggio parametri creazione rete
            file_hyper = os.path.join(directory_run, 'hyper.json')
            json.dump(_param_net, open(file_hyper, 'w'))
            
            # Salvataggio sets utilizzati
            file_train = os.path.join(directory_run, 'train_norm_set_'+startime_run+'.csv')
            file_valid_norm = os.path.join(directory_run, 'valid_norm_set_'+startime_run+'.csv')
            np.savetxt(file_train, x_train_norm, delimiter=',')
            np.savetxt(file_valid_norm, x_valid_norm, delimiter=',')
                    
            # Plot history
            train_loss_plot(history, True, os.path.join(directory_run, 'train_test_loss.png'))
            
            # Salvataggio rete neurale
            name_net = 'net_'+startime_run+'.h5'        
            print("Saving net...name:{}".format(name_net))
            file_net = os.path.join(directory_run, name_net)
            autoencoder_model.save(file_net)
        
        gen_net +=1


# **Parte Hyperopt**



import code_test.lib_nn_prediction_analysis as lib_nn_prediction_analysis

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import pickle


# Configurazione:




default_autoencoder_hyperopt = {
    'epochs': 50, # totale epoche addestramento
    'batch_size': 64, # dimensione batch
    'shuffle': True, # abilitazione shuffle .fit
    'HP_METRIC': 'accuracy',
    # iperparametri hyperopt support
    'num_layers': hp.choice('num_layers', np.arange(2, 8, 2)), # numero layer come iperparametro rete
    'num_unit_1': hp.choice('num_unit_1', np.arange(30, 80, 5)), # numero neuroni di un layer
    'num_unit_2': hp.choice('num_unit_2', np.arange(50, 100, 5)),
    'num_unit_3': hp.choice('num_unit_3', np.arange(75, 120, 5)),
    'num_unit_4': hp.choice('num_unit_4', np.arange(90, 150, 10)),
    'num_unit_5': hp.choice('num_unit_5', np.arange(120, 170, 10)),
    'num_unit_6': hp.choice('num_unit_6', np.arange(150, 200, 25)),
    'actv_func': hp.choice('actv_func', ['relu', 'elu']), # funzioni di attivazione
    'actv_func_out': 'relu', # funzione di attivazione uscita
    'loss_func': hp.choice('loss_func', ['mse', 'mae']), # funzione di perdita
    'l1_reg': hp.loguniform('l1_reg', -9, -7), # regolarizzatore
    'learn_rate_opt': hp.loguniform('learn_rate_opt', -7, -6), # learning rate
    'optimizer': hp.choice('optimizer', ['adam', 'adadelta']), # ottimizzatore
    'drop_enabled': hp.choice('drop_enabled', [True, False]), # abilitazione layer Dropout
    'drop_factor': hp.loguniform('drop_factor', -2.3, -1.2) # fattore di Dropout
}


# Rete neurale:




def autoencoder_hyperopt(_n_features, _hparams = default_autoencoder_hyperopt):
    """
    Creazione struttura autoencoder Keras parametrica.
    
    :param _n_features: int
        numero input rete neurale. Configurazione tensore principale
    :param _hparams: dict
        parametri per la configurazione della rete
    
    :return: Model
        autoencoder creato. Keras model
    """
    input_layer = Input(shape=(_n_features,))
    for layer in range(1,int(_hparams['num_layers'])+1):
        
        hidden = Dense(units=int(_hparams['num_unit_'+str(layer)]), activation=_hparams['actv_func'], activity_regularizer=regularizers.l1(_hparams['l1_reg']))(input_layer if layer == 1 else hidden)
        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)
    
    for layer in reversed(range(1,int(_hparams['num_layers'])+1)):
        hidden = Dense(units=int(_hparams['num_unit_'+str(layer)]), activation=_hparams['actv_func'], activity_regularizer=regularizers.l1(_hparams['l1_reg']))(hidden)
        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)
        
    output_layer = Dense(_n_features, activation=_hparams['actv_func_out'])(hidden)
    
    autoencoder = Model(input_layer, output_layer)
    
    if _hparams['optimizer'] == 'adadelta':
        opt_net = Adadelta(_hparams['learn_rate_opt'], rho=0.95)
    elif _hparams['optimizer'] == 'adam':
        opt_net = Adam(_hparams['learn_rate_opt'], beta_1=0.9, beta_2=0.999, amsgrad=False)
    else:
        opt_net = Adam(_hparams['learn_rate_opt'], beta_1=0.9, beta_2=0.999, amsgrad=False)
            
    autoencoder.compile(optimizer=opt_net, loss=_hparams['loss_func'])
    
    return autoencoder





def run_hyperopt_acc_with_anom(_load_dataset, _param_net = default_autoencoder_hyperopt, _directory_save = 'default_nets', _save_temp_net = True,
                          _df_norm = '', _data_anom = '', x_train_norm = '', x_test_norm = '', x_valid_norm = '',
                          _max_eval_hopt = 10, _algorithm_hopt = tpe.suggest):
    """
    DEPRECATED 
    Versione facilitata(solo prime versioni PRIVATE DATA - NOT LOADED reale)
    
    Generazione rete neurale.
    Run versione hyperopt con accuratezza e utilizzando nella valutazione dataset anomalo. 
    
    :param _load_dataset : bool
        True se si desidera utilizzare una partizione di train/test/validation già esistente. 
        False se si desidera procedere con uno split randomico.
        Entrambe le forme utilizzano dati già normalizzati
    :param _param_net : dict
        parametri di configurazione rete. Per creazione, addestramento... 
    :param _directory_save : str
        percorso della cartella di destinazione per il salvataggio delle reti
    :param _save_temp_net : bool
        True se si desidera salvare tutta la cartella con i relativi files per il recupero della rete neurale generata.
        False altrimenti.
    :param _df_norm : Dataframe
        punti normali già normalizzati su cui eseguire split in train e test/validation. Utilizzato se _load_dataset è FALSE
    :param _data_anom : numpy
        punti anomali già normalizzati. 
    :param x_train_norm : numpy array
        punti normali già normalizzati e pre caricati di addestramento. Utilizzato se _load_dataset è TRUE
    :param x_test_norm : numpy array
        punti normali già normalizzati e pre caricati di testing. Utilizzato se _load_dataset è TRUE
    :param x_valid_norm : numpy array
        punti normali già normalizzati e pre caricati di validazione. Utilizzato se _load_dataset è TRUE
    :param _max_eval_hopt: int
        numero massimo valutazioni hypeopt
    :param _algorithm_hopt:
        algoritmo Hyperopt da utilizzare
    """
    if not _load_dataset:
        # necessito split
        print('Eseguo split punti normali, creazione train, test, validation sets')
        x_train_norm, x_test_norm, x_valid_norm, _ = split_train_test_valid(_df_norm)
    else:
        print('Saranno utilizzati dataset caricati')
        # utilizzo parametri metodo

    n_features = x_train_norm.shape[1]

    startime_run = str(time.time())
    if not os.path.exists(_directory_save):
        print("Creazione directory {}".format(_directory_save))
        os.mkdir(_directory_save)
    
    def run_net(params):
        startime_run = str(time.time())
        
        # Creo nuova struttura autoencoder
        autoencoder_model = autoencoder_hyperopt(n_features, params)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=1e-5) 
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)
        
        history = autoencoder_model.fit(x_train_norm, x_train_norm, epochs=params['epochs'], 
                validation_data=(x_valid_norm, x_valid_norm),
                shuffle=params['shuffle'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0)
        
        test_acc, class_report, error_threshold = lib_nn_prediction_analysis.evaluate_acc_model_with_anom_dataset(autoencoder_model, 
                                                                                        x_train_norm, x_test_norm, _data_anom,
                                                                                        params['loss_func'])
        
        if _save_temp_net:
            directory_run = os.path.join(_directory_save, startime_run)
            # Creazione cartella della specifica rete neurale generata
            os.mkdir(directory_run)
            
            # Salvataggio struttura
            file_summary = os.path.join(directory_run, 'summary.txt')
            with open(file_summary,'w') as fh:
                autoencoder_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            # Salvataggio parametri creazione rete. json problemi con serializzazione hp
            file_hyper = os.path.join(directory_run, 'hyper.pkl')
            pickle.dump(params, open(file_hyper, 'wb')) # per read pickle.load(open(file_hyper, 'rb'))
            
            # Salvataggio report
            file_report = os.path.join(directory_run, 'report.txt')
            with open(file_report,'w') as fh:
                fh.write(class_report)
            
            # Salvataggio sets utilizzati
            file_train = os.path.join(directory_run, 'train_norm_set_'+startime_run+'.csv')
            file_test_norm = os.path.join(directory_run, 'test_norm_set_'+startime_run+'.csv')
            file_valid_norm = os.path.join(directory_run, 'valid_norm_set_'+startime_run+'.csv')
            np.savetxt(file_train, x_train_norm, delimiter=',')
            np.savetxt(file_test_norm, x_test_norm, delimiter=',')
            np.savetxt(file_valid_norm, x_valid_norm, delimiter=',')
                    
            # Plot history
            train_loss_plot(history, True, os.path.join(directory_run, 'train_test_loss.png'))
            
            # Salvataggio rete neurale
            name_net = 'net_'+startime_run+'acc_'+str(test_acc)+'.h5'
            print("Saving net...name:{}".format(name_net))
            file_net = os.path.join(directory_run, name_net)
            autoencoder_model.save(file_net)
        
        return {'loss': -test_acc, 'status': STATUS_OK}
    
    best = fmin(run_net, _param_net, algo=_algorithm_hopt, max_evals=_max_eval_hopt)





def run_hyperopt_semisupervised(_load_dataset, _discriminator = 'avg_err', _param_net = default_autoencoder_hyperopt, _directory_save = 'default_nets', _save_temp_net = True,
                          _df_norm = '', x_train_norm = '', x_test_norm = '', x_valid_norm = '',
                          _max_eval_hopt = 10, _algorithm_hopt = tpe.suggest):
    """
    Generazione rete neurale.
    
    Run versione hyperopt completamente semi-supervised.
    
    Possibile abilitare la ricerca della rete migliore 
    tramite errore di predizione medio('avg_err') o tramite accuratezza('acc') sul testset normale
    
    :param _load_dataset : bool
        True se si desidera utilizzare una partizione di train/test/validation già esistente. 
        False se si desidera procedere con uno split randomico.
        Entrambe le forme utilizzano dati già normalizzati
    :param _discriminator: str
        discrimanatore rete migliore. 
        'avg_err' per errore di predizione medio basso sul testset normale
        'acc' per accuratezza sul testset normale
    :param _param_net : dict
        parametri di configurazione rete. Per creazione, addestramento... 
    :param _directory_save : str
        percorso della cartella di destinazione per il salvataggio delle reti
    :param _save_temp_net : bool
        True se si desidera salvare tutta la cartella con i relativi files per il recupero della rete neurale generata.
        False altrimenti.
    :param _df_norm : Dataframe
        punti normali già normalizzati su cui eseguire split in train e test/validation. Utilizzato se _load_dataset è FALSE
    :param x_train_norm : numpy array
        punti normali già normalizzati e pre caricati di addestramento. Utilizzato se _load_dataset è TRUE
    :param x_test_norm : numpy array
        punti normali già normalizzati e pre caricati di testing. Utilizzato se _load_dataset è TRUE
    :param x_valid_norm : numpy array
        punti normali già normalizzati e pre caricati di validazione. Utilizzato se _load_dataset è TRUE
    :param _max_eval_hopt: int
        numero massimo valutazioni hypeopt
    :param _algorithm_hopt:
        algoritmo Hyperopt da utilizzare
    """
            
    if not _load_dataset:
        # necessito split
        print('Eseguo split punti normali, creazione train, test, validation sets')
        x_train_norm, x_test_norm, x_valid_norm, _ = split_train_test_valid(_df_norm)
    else:
        print('Saranno utilizzati dataset caricati')
        # utilizzo parametri metodo

    n_features = x_train_norm.shape[1]

    startime_run = str(time.time())
    if not os.path.exists(_directory_save):
        print("Creazione directory {}".format(_directory_save))
        os.mkdir(_directory_save)
        
    def run_net(params):
        startime_run = str(time.time())
        
        # Creo nuova struttura autoencoder
        autoencoder_model = autoencoder_hyperopt(n_features, params)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=1e-5) 
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)
        
        history = autoencoder_model.fit(x_train_norm, x_train_norm, epochs=params['epochs'], 
                validation_data=(x_valid_norm, x_valid_norm),
                shuffle=params['shuffle'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0)
        
        if _discriminator == 'acc':
            test_acc, class_report, error_threshold = lib_nn_prediction_analysis.evaluate_acc_model_with_norm_dataset(autoencoder_model, x_train_norm, x_test_norm,
                                                                                                                      params['loss_func'])
        else:
            #default 'avg_err'
            error_threshold, avg_error = lib_nn_prediction_analysis.evaluate_avg_error_model_semi_supervised(autoencoder_model, x_train_norm, x_test_norm,
                                                                                                             params['loss_func'])
        
        if _save_temp_net:
            directory_run = os.path.join(_directory_save, startime_run)
            # Creazione cartella della specifica rete neurale generata
            os.mkdir(directory_run)
            
            # Salvataggio struttura
            file_summary = os.path.join(directory_run, 'summary.txt')
            with open(file_summary,'w') as fh:
                autoencoder_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            # Salvataggio parametri creazione rete. json problemi con serializzazione hp
            file_hyper = os.path.join(directory_run, 'hyper.pkl')
            pickle.dump(params, open(file_hyper, 'wb')) # per read pickle.load(open(file_hyper, 'rb'))
            
            if _discriminator == 'acc':
                # Salvataggio report
                file_report = os.path.join(directory_run, 'report.txt')
                with open(file_report,'w') as fh:
                    fh.write(class_report)
            
            # Salvataggio sets utilizzati
            file_train = os.path.join(directory_run, 'train_norm_set_'+startime_run+'.csv')
            file_test_norm = os.path.join(directory_run, 'test_norm_set_'+startime_run+'.csv')
            file_valid_norm = os.path.join(directory_run, 'valid_norm_set_'+startime_run+'.csv')
            np.savetxt(file_train, x_train_norm, delimiter=',')
            np.savetxt(file_test_norm, x_test_norm, delimiter=',')
            np.savetxt(file_valid_norm, x_valid_norm, delimiter=',')
                    
            # Plot history
            train_loss_plot(history, True, os.path.join(directory_run, 'train_test_loss.png'))
            
            # Salvataggio rete neurale
            name_net = 'net_'+startime_run+'disc_'+_discriminator+'.h5'
            print("Saving net...name:{}".format(name_net))
            file_net = os.path.join(directory_run, name_net)
            autoencoder_model.save(file_net)
        
        if _discriminator == 'acc':
            return {'loss': -test_acc, 'status': STATUS_OK}
        else:
            #default 'avg_err'
            return {'loss': avg_error, 'status': STATUS_OK}  
        
    best = fmin(run_net, _param_net, algo=_algorithm_hopt, max_evals=_max_eval_hopt)
    













