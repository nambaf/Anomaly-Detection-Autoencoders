#!/usr/bin/env python
# coding: utf-8

# # Caso di studio: PRIVATE DATA - NOT LOADED 
# 
# Cartella di riferimento: \PRIVATE DATA - NOT LOADED
# 
# File dataset completo:  PRIVATE DATA - NOT LOADED
# 
# **.csv all'interno della cartella di riferimento**:
#     - se [nome_anomalia].csv: dati che fanno riferimento agli indici start e end secondo tabella sotto
#     - se norm_data_blind_1[_normalized].csv: punti di train. Con e senza normalizzazione
#     - se test_data_blind_1[_normalized].csv: punti di test, resto delle mensilità. Con e senza normalizzazione
#     - se test_blind_1_subset_[1...11]_normalized.csv: punti di test già normalizzati partizionati per indici concordati
#  
# Composizione dataset:
# 


# Librerie:



import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import sys


# Percorso:



dir_artf = '..\\..\\..\\dataset\\Artificiali\\PRIVATE DATA - NOT LOADED\\'

subset_dir = dir_artf + 'Generated\\PRIVATE DATA - NOT LOADED'
PRIVATE DATA - NOT LOADED = dir_artf + 'PRIVATE DATA - NOT LOADED.csv'


# Generazione Dataframe:

PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED)
# Analisi


print('Dimensionalità chp:',PRIVATE DATA - NOT LOADED.shape)


# I primi due elementi
PRIVATE DATA - NOT LOADED.head(2)


# Andamento temperatura esterna:


PRIVATE DATA - NOT LOADED.plot(x='OLD_INDEX', y='Text', figsize=(20,10))


# **Creazione variabili artificiali gradienti, carico**

from lib_preprocessing import calc_gradient, calc_gradient_avg, chp_creation_ele_perf_ratio


# Il carico massimo della macchina deve essere calcolato dal valore massimo di potenza elettrica rilevato dai dati normali:


maxvalue_actpower_training = 0
print(maxvalue_actpower_training)




# **Dataset normale e test**
# 
# I primi 178560 elementi costituiscono i punti normali su cui addestrare la rete. 
# 
# Il dataset di test sarà partizionato ad anomalie e ad indici concordati per una più facile visualizzazione e analisi.


"""
Creazione e salvataggio datasets CHP

norm_data : Dataframe
    Dataframe dei punti normali
test_data_all : Dataframe
    Dataframe dei punti di test
"""

norm_data = PRIVATE DATA - NOT LOADED.head(178560)
test_data_all = PRIVATE DATA - NOT LOADED.loc[178560:PRIVATE DATA - NOT LOADED.shape[0]]

# Elimino colonna target e index dai punti normali - 0
norm_data = norm_data.drop(['OLD_INDEX', 'Anom'], axis=1)

# Nome dataset 
norm_data_name = 'PRIVATE DATA - NOT LOADED'
test_data_name = 'PRIVATE DATA - NOT LOADED'

norm_data.to_csv(os.path.join(subset_dir, norm_data_name + '.csv'),index=False)
test_data_all.to_csv(os.path.join(subset_dir, test_data_name + '.csv'),index=False)




print('Ultimo salvataggio: ',norm_data_name, test_data_name)


# **Normalizzazione dei dati**


# Librerie
from sklearn.preprocessing import MinMaxScaler
from lib_preprocessing import normalize_min_max


dir_data_chp = '..\\..\\..\\dataset\\Artificiali\\Datasets_blind\\Generated\\PRIVATE DATA - NOT LOADED'

norm_file_name = 'PRIVATE DATA - NOT LOADED'
test_file_name = 'PRIVATE DATA - NOT LOADED'


df_norm = pd.read_csv(os.path.join(dir_data_chp,norm_file_name + '.csv'))
df_test = pd.read_csv(os.path.join(dir_data_chp,test_file_name + '.csv'))


print('Massimo carico nel training:',max(df_norm['PRIVATE DATA - NOT LOADED']))



print('Massimo carico nel test, tra gli stati normali:',max(df_test[df_test['Anom'] == 0]['PRIVATE DATA - NOT LOADED']))


# **Normalizzazione temperatura esterna**
# 
# Normalizzazione dei dati della temperatura scegliendo un dominio fissato la minima e la massimo PRIVATE DATA - NOT LOADED
# 
# Dominio: Minima - 10 gradi e Massima 40 gradi



print('Valori minimi:',min(df_norm['Text']),min(df_test['Text']))
print('Valori massimi:',max(df_norm['Text']),max(df_test['Text']))



df_norm['Text'].plot()



df_test['Text'].plot()



# Normalizzazione temperatura nei range [-10,40]
temp_norm_normalized = normalize_min_max(df_norm['Text'],-10,40)
temp_test_normalized = normalize_min_max(df_test['Text'],-10,40)


# #### Normalizzazione altri parametri:
# 
# Ipotesi 1: fissare dei range di dominio per gli attributi **indipendentemente** dal tempo
# 
# Ipotesi 2: accettare come normalizzazione i range possibili con una % di incertezza nell'arco dell'anno. Ipotesi ammessa se accettato aggiornamento scaler negli anni.
# 
# Ipotesi 3: applicare MinMaxScaler dai dati di training e recuperarlo per scalare il testset. Problematico se il training non è adeguatamente completo(stati normali nel test che escono da dominio causa tara su training)
# 
# Percorso Ipotesi 2.


df_norm.head(1)



# PRIVATE DATA - NOT LOADED
column = 'PRIVATE DATA - NOT LOADED'
print('Valori minimi {}:'.format(column),min(df_norm[column]),min(df_test[df_test['Anom'] == 0][column]))
print('Valori massimi {}:'.format(column),max(df_norm[column]),max(df_test[df_test['Anom'] == 0][column]))
diff_norm_normalized = normalize_min_max(df_norm[column],-5,10)
diff_test_normalized = normalize_min_max(df_test[column],-5,10)


# I valori nei test possono essere fuori range per via del non filtro sui soli dati normali
column = 'PRIVATE DATA - NOT LOADED'
print('Valori minimi {} normalizzati:'.format(column),min(temp_norm_normalized),min(temp_test_normalized))
print('Valori massimi {} normalizzati:'.format(column),max(temp_norm_normalized),max(temp_test_normalized))


list_col_min_max = ['PRIVATE DATA - NOT LOADED']

# Utilizzare lo stesso scaler per l'intero Dataset!!!
scaler = MinMaxScaler().fit(df_norm[list_col_min_max].values)
# Iniziale per normalizzazione gradienti
df_norm[list_col_min_max] = scaler.transform(df_norm[list_col_min_max].values)
df_test[list_col_min_max] = scaler.transform(df_test[list_col_min_max].values)


# Applicazione normalizzazioni:


df_norm['PRIVATE DATA - NOT LOADED'] = temp_norm_normalized
df_test['PRIVATE DATA - NOT LOADED'] = temp_test_normalized


# Salvataggio dataset normalizzati:

# Nome dataset
norm_data_name = 'PRIVATE DATA - NOT LOADED'
test_data_name = 'PRIVATE DATA - NOT LOADED'

df_norm.to_csv(os.path.join(dir_data_chp, norm_data_name + '_normalized.csv'),index=False)
df_test.to_csv(os.path.join(dir_data_chp, test_data_name + '_normalized.csv'),index=False)



print('Ultimo salvataggio: ',norm_data_name, test_data_name)


# Risultato:


print('Dimensionalità dataset train:',df_norm.shape)




print('Dimensionalità dataset test:',df_test.shape)


# **Generazione subset normalizzati**



df_test = pd.read_csv(os.path.join(dir_data_chp,test_file_name + '_normalized.csv'))