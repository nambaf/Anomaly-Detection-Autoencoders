#!/usr/bin/env python
# coding: utf-8

# # Generazione Dataframe da file

# Per una più facile manipolazione dei files è stato utilizzato concetto di [Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) della libreria [Pandas](https://pandas.pydata.org/).
# Per ogni cartella/file ricevuto di dati di impianti/macchine differenti si è creato un dataset di tipo Dataframe.
# Da ogni dataset di tipo Dataframe si sono creati diversi Dataframe per ogni macchina descritta al suo interno.  
# 
# **Questo file è possibile anche non utilizzarlo. Si possono usare direttamente i Dataframe creati e salvati in formato .csv**
# PRIVATE DATA - NOT LOADED
#    
# ##### Edit: Marzo 2020
# 
# Librerie utilizzate:




import sys
import os
import pandas as pd
import csv


# Percorsi files:




DIR_2013_FIRST = 'PRIVATE DATA - NOT LOADED'
DIR_2017_PONTLAB = 'PRIVATE DATA - NOT LOADED'
DIR_HOTEL_ETR = 'PRIVATE DATA - NOT LOADED'
DIR_PONTLAB_15MIN_TEMPERATURE = 'PRIVATE DATA - NOT LOADED'
FILE_ARTFICIAL = 'PRIVATE DATA - NOT LOADED'
FILE_ARTFICIAL_NEW = 'PRIVATE DATA - NOT LOADED' # versione aggiornata Marzo 2020
DIR_PROSPUMB = 'PRIVATE DATA - NOT LOADED'


# ## Generazione Dataframe datasets




def read_files(_files_name, _archive = 1, _skip_row = 0, _separator = ';'):
    """
    Lettura file xls/csv e creazione Dataframe.
    
    :param _files_name : str
        Un file o una lista di xls/csv da leggere
    :param _archive : int
        Numero pagina o foglio xls. Dipende da file 
    :param _skip_row : int 
        Numero di righe da saltare per lettura file xls. Dipende da file
    :param _separator : char
        Separatore del file csv. Dipende da file
    
    :return: Dataframe
        Contiene i dati importati
    """
    if not isinstance(_files_name, list):
        # file
        if _files_name.endswith('.xls'):
            res = pd.read_excel(open(_files_name,'rb'),sheet_name=_archive,skiprows=_skip_row)
        else:
            res = pd.read_csv(open(_files_name,'rb'),sep=_separator)
        return res
    else:
        # list
        res = pd.DataFrame()
        for file in _files_name:
            if file.endswith('.xls'):
                df = pd.read_excel(open(file,'rb'),sheet_name=_archive,skiprows=_skip_row)
            else:
                df = pd.read_csv(open(file,'rb'),sep=_separator)
            res = pd.concat([res, df], axis=0, sort=False)
            res.reset_index(inplace=True, drop=True)
        return res


# Impianto PontLab ha ulteriore divisione in dati elettrici e termici




# Decommentare in base ai file da importare. In caso di cartelle con più file da leggere
ELE_DATA = PRIVATE DATA - NOT LOADED + '\\Elettrico'
TER_DATA = PRIVATE DATA - NOT LOADED + '\\Termico'


# Gli altri dati forniti sono in un'unica cartella




# Recupero tutti i files
folder = DIR_2013_FIRST # cartella o file per lettura .xls o .csv di
archive = 2 # pagina o foglio .xls. Dipende da file
skip_row = 11 # numero di righe da saltare per lettura file. Dipende da file
sepator = ';' # separatore utilizzato nel file da importare. Dipende da file

# Nome file output Dataframe. Ogni dataset ha il suo Dataframe e il suo file per la generazione dei dispositivi.
# !Vedere glossario files!
output_file = 'nome_dataset.csv'





# Da utilizzare in caso di cartelle con più file da leggere
#onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


# Creazione Dataframe da file importati




# se file .xls
#result = read_files(onlyfiles,archive, skip_row)

# se file .csv
result = read_files(onlyfiles,_separator = ',')





print('Le features del Dataframe creato:\n',result.columns)


# Cambio nome features:




columns_to_rename = {"Unnamed: 0": "LEGALE_SOLARE", "Unnamed: 1": "DATE_TIME"}

result = result.rename(index=str, columns=columns_to_rename)


# **In alcuni files necessaria un'ulteriore fase di filtraggio**




# Esempio procedimento usato per PRIVATE DATA - NOT LOADED. 
# Dati della temperatura a 15 minuti da estendere al resto dei punti. Recupero e merge Dataframes 
# df_to_add : Dataframe che contiene le righe a 15 secondi con i dati della temperatura a 15 minuti estesi
# result : Dataframe che contiene i dati di CHP PontLab a 15 minuti  
# union: Dataframe che contiene i dati CHP con i dati di temperatura estesi
temp_data_subset = ["DATE_TIME","TEMPERATURE"]
result = result.loc[:,temp_data_subset]
result = result.dropna()
result = result.reset_index(drop=True)
row_list = []
for index, row in result.iterrows():
    actual_date = row['DATE_TIME']
    actual_temp = row['TEMPERATURE']
    # creo snapshot riga ogni 15 secondi. Ho dato temperatura a 15 minuti e lo replico per i successivi 14 minuti e 45 secondi. Concordato
    for i in range(15,(840 + 60),15):
        date_to_add = actual_date + pd.Timedelta(seconds=i)
        constant_temp = actual_temp
        row_list.append({'DATE_TIME':date_to_add,'TEMPERATURE':constant_temp})

df_to_add = pd.DataFrame(row_list, columns=['DATE_TIME','TEMPERATURE'])
df_to_add['DATE_TIME'] = pd.to_datetime(df_to_add['DATE_TIME'])
result['DATE_TIME'] = pd.to_datetime(result['DATE_TIME'])

union = result.append(df_to_add)
union = union.sort_values(by='DATE_TIME')
union = union.reset_index(drop=True)
union.to_csv(output_file,index=False)


# Salvataggio Dataframe:




# Salvo il file
result.to_csv(output_file,index=False)


# ## Generazione Dataframe dataset artificiali

# I dataset per le reti neurali mantengolo la seguente struttura:
#     - dati normali: i primi 178560 elementi
#     - dati di test: tutti gli altri elementi. Punti normali e anomali
#     
# Il primo dataset artificiale simula i dati di un PRIVATE DATA - NOT LOADED.




# Primo procedimento usato per FILE_ARTFICIAL.
# Successivamente è stato standardizzato come per i dataset blind post Marzo 2020
# Creazione e salvataggio due Dataframe e rinominazione colonne. 
# norm_data : Dataframe punti normali CHP
# test_data_all : Dataframe punti di test CHP
rebuild_dir = '..\\dataset\\Artificiali\\Generated'
df_art = pd.read_csv(rebuild_dir + '\\PRIVATE DATA - NOT LOADED.csv',sep=';')
df_art = df_art.drop(columns=['Unnamed: 0'],axis=1)

columns_to_rename = {'PRIVATE DATA - NOT LOADED':'PRIVATE DATA - NOT LOADED'}
df_art = df_art.rename(index=str,columns=columns_to_rename)
norm_data = df_art.head(178560)
norm_data.to_csv(os.path.join(rebuild_dir, 'PRIVATE DATA - NOT LOADED.csv'),index=False)
test_data_all = df_art.iloc[178561:2102454]
# In alcune versioni esistenza punti NaN nei test da azzerare. Concordato
test_data_all['PRIVATE DATA - NOT LOADED'] = test_data_all['PRIVATE DATA - NOT LOADED'].where(test_data_all['PRIVATE DATA - NOT LOADED'].notnull(), 0)
test_data_all['PRIVATE DATA - NOT LOADED'] = test_data_all['PRIVATE DATA - NOT LOADED'].where(test_data_all['PRIVATE DATA - NOT LOADED'].notnull(), 0)
test_data_all.to_csv(os.path.join(rebuild_dir, 'chp_artificial_test_new.csv'))


# ## Edit: lettura datasets blind da Marzo 2020
# 
# Alcuni dei file importati necessitano trasformazione( per esempio creazione riga delle colonne mancanti)
# 
# Molteplici Dataframe per ogni dataset blind:
# 1. (OPZIONALE) *NOME_DATASET*_rebuild: versione necessaria per aggiunta labels colonne Dataframe. Se presente utilizzata questa versione per creazione Dataframe punti normali e Dataframe punti di test
# 2. *NOME_DATASET*: Dataframe dispositivo utilizzato per creazione Dataframe punti normali e Dataframe punti di test
# 3. *NOME_DATASET*_labeled: Dataframe dispositivo con aggiunta colonne target per identificazione dello stato( normale o anomalo)

# **Prima parte: se non si dispone del file con le labels**
# [OPZIONALE]




# Percorso cartella datasets blind
DIR_BLIND = 'PRIVATE DATA - NOT LOADED'





FILE_BLIND_1 = 'PRIVATE DATA - NOT LOADED' # CHP simulato stessa tipologia dati PRIVATE DATA - NOT LOADED
FILE_BLIND_2 = 'PRIVATE DATA - NOT LOADED' # CHP simulato taglia maggiore





# Lettura file e aggiunta labels 
with open(os.path.join(DIR_BLIND, FILE_BLIND_2 +'.csv'),newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open(os.path.join(DIR_BLIND, FILE_BLIND_2 +'_rebuild.csv'),'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['PRIVATE DATA - NOT LOADED'])
    w.writerows(data)





# Ogni Dataframe(punti normali e di test) viene salvato nella propria cartella 
subset_dir = 'PRIVATE DATA - NOT LOADED'





# Aggiunta suffisso numerico ad ogni Dataframe per ulteriore controllo evitando errori nella copia di scripts 
df_art = pd.read_csv(os.path.join(DIR_BLIND, FILE_BLIND_2 +'_rebuild.csv'))
norm_data = df_art.head(178560)
norm_data.to_csv(os.path.join(subset_dir, 'PRIVATE DATA - NOT LOADED.csv'),index=False)
test_data_all = df_art.iloc[178561:2102454]
test_data_all.to_csv(os.path.join(subset_dir, 'PRIVATE DATA - NOT LOADED.csv'))


# **Seconda parte: se si dispone del file con le labels**
# 
# Sovrascrivo file fornito per mantenere standard script.




FILE_BLIND_1 = 'PRIVATE DATA - NOT LOADED' 
FILE_BLIND_2 = 'PRIVATE DATA - NOT LOADED'

FILE_ARTF_GENNAIO = 'PRIVATE DATA - NOT LOADED'





rebuild_dir = 'PRIVATE DATA - NOT LOADED'
rebuild_dir_gennaio = 'PRIVATE DATA - NOT LOADED'
# se si avvia questa istruzione post sostituzione file, rimuovere parte separatore ;
df_art = pd.read_csv(os.path.join(rebuild_dir_gennaio, FILE_ARTF_GENNAIO +'.csv'),sep=';')





# Esempio riga
df_art.head(1)





columns_to_rename = {'PRIVATE DATA - NOT LOADED':'PRIVATE DATA - NOT LOADED'}
df_art = df_art.rename(index=str,columns=columns_to_rename)

df_art.to_csv(os.path.join(rebuild_dir_gennaio, FILE_ARTF_GENNAIO +'.csv'),index=False)



