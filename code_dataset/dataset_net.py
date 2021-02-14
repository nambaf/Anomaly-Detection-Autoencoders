#!/usr/bin/env python
# coding: utf-8

# # Dataset 

# **Cartella di riferimento: dataset\\PRIVATE DATA - NOT LOADED**
# 
# File dataset: PRIVATE DATA - NOT LOADED
# 
# Dati relativi a un PRIVATE DATA - NOT LOADED
# 
# Nella sottocartella \Generated sono salvati i Dataframe di ogni dispositivo



# Librerie
import pandas as pd




# Files di riferimento
dir_files = '..\\..\\dataset\\PRIVATE DATA - NOT LOADED\\Generated\\'
PRIVATE DATA - NOT LOADED = dir_files + 'PRIVATE DATA - NOT LOADED.csv'
PRIVATE DATA - NOT LOADED = dir_files + 'PRIVATE DATA - NOT LOADED.csv'

# File generati
output_PRIVATE DATA - NOT LOADED = dir_files + 'PRIVATE DATA - NOT LOADED.csv'
output_PRIVATE DATA - NOT LOADED = dir_files + 'PRIVATE DATA - NOT LOADED.csv'
output_PRIVATE DATA - NOT LOADED = dir_files + 'PRIVATE DATA - NOT LOADED.csv'


# Lettura
PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED, dtype={'PRIVATE DATA - NOT LOADED':str})
PRIVATE DATA - NOT LOADED = pd.read_csv(PRIVATE DATA - NOT LOADED, dtype={'PRIVATE DATA - NOT LOADED':str})


print('Features PRIVATE DATA - NOT LOADED =',df_ele.columns)
print('Features PRIVATE DATA - NOT LOADED =',df_ter.columns)


# Eliminazione NaN e reset index



# Merge features elettriche e termiche per creazione Dataframe dispositivi 
PRIVATE DATA - NOT LOADED = pd.merge(PRIVATE DATA - NOT LOADED, PRIVATE DATA - NOT LOADED, on='DATE_TIME', how='inner')
PRIVATE DATA - NOT LOADED = pd.merge(PRIVATE DATA - NOT LOADED, PRIVATE DATA - NOT LOADED, on='DATE_TIME', how='inner')
# non necessario per abs. Solo dati termici


# Salvataggio Dataframe
PRIVATE DATA - NOT LOADED.to_csv(output_chp,index=False)
PRIVATE DATA - NOT LOADED.to_csv(output_chil,index=False)
PRIVATE DATA - NOT LOADED.to_csv(output_abs,index=False)

