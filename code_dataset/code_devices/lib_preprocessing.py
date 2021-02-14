#!/usr/bin/env python
# coding: utf-8

# # Libreria utilizzata in fase di preprocessing

# Utilizzato per creazione variabile artificiale gradiente(ex.PRIVATE DATA - NOT LOADED) come differenza istanti temporali. Necessita ordinamento temporale.
# 
# EXPENSIVE - ONLY OFFLINE




def calc_gradient(_data, _row, _column_name, _ratio = 1):
    """
    Ritorna il valore gradiente(oscillazione) di un attributo rispetto un certo intervallo di tempo.
    Utilizzato per creare un attributo temporale artificiale per il modello neurale.
    Necessita _data, intero dataset, per recupero delle row successive, no controllo temporale successivo 
    ma solo tramite shift row.
    
    :param _data: Dataframe
        l'intero dataset, no shiftato ma ordinato temporalmente
    :param _row: entry Dataframe
        riga del Dataframe da considerare per il calcolo del gradiente
    :param _column_name: str
        il nome della colonna, attributo di cui calcolare il gradiente
    :param _ratio: int
        la posizione riga precedente da considerare per il calcolo del gradiente
    
    :return: float
        il valore del gradiente
    """
    actual_value = _row[_column_name]
    index_row = _data.index.get_loc(_row.name)
    if (index_row - _ratio) >= 0:
        value_after_ratio = _data.iloc[index_row - _ratio][_column_name]
        gradient = abs(actual_value-value_after_ratio)
        return gradient
    else:
        #non ho abbastanza info, fisso il gradiente nullo
        gradient = 0
        return gradient


# Utilizzato per creazione variabile artificiale gradiente(ex.PRIVATE DATA - NOT LOADED) come media istanti temporali precedenti. Necessita ordinamento temporale.
# 
# EXPENSIVE - ONLY OFFLINE




def calc_gradient_avg(_data, _row, _column_name, _ratio = 4):
    """
    Ritorna il valore gradiente(oscillazione) di un attributo rispetto un certo intervallo di tempo 
    considerando la media dei valori precedenti.
    Utilizzato per creare un attributo temporale artificiale per il modello neurale.
    Necessita _data, intero dataset, per recupero delle row successive, no controllo temporale successivo 
    ma solo tramite shift row.
    
    :param _data: Dataframe
        l'intero dataset, no shiftato ma ordinato temporalmente
    :param _row: entry Dataframe
        riga del Dataframe da considerare per il calcolo del gradiente
    :param _column_name: str
        il nome della colonna, attributo di cui calcolare il gradiente
    :param _ratio: int
        le posizioni righe precedente da considerare per il calcolo del gradiente
        
    :return: float
        il valore del gradiente
    """
    actual_value = _row[_column_name]
    index_row = _data.index.get_loc(_row.name)
    if (index_row - _ratio) >= 0:
        value_after_ratio = 0
        for prec_row in range(index_row - _ratio, index_row):
            value_after_ratio += _data.iloc[prec_row][_column_name]
        value_after_ratio /= _ratio
        gradient = abs(actual_value-value_after_ratio)
        return gradient
    else:
        #non ho abbastanza info allora il gradiente lo considero nullo
        gradient = 0
        return gradient


# Utilizzato per creazione feature artificiale carico PRIVATE DATA - NOT LOADED. Info datasheet:



# Per migliorare i plot




def connectpoints(_ax, _x,_y):
    """
    Creazione retta grafica
    
    :param _ax: Axes
        asse per plot    
    :param _x: list
        coordinate x delle rette da creare       
    :param _y: list
        coordinate y delle rette da creare
    """
    for index, value in enumerate(_x[:-1]):
        x1, x2 = _x[index], _x[index+1]
        y1, y2 = _y[index], _y[index+1]
        _ax.plot([x1,x2],[y1,y2],'g-')


# Filtro grafico ma applicato poi a Dataframe con curve di upperbound e lowerbound definite.
# Rimangono i soli punti all'interno delle curve definite. Utilizzato per individuazione comportamento normale dispositivo successivamente la definizione dei range accettabili.




import numpy as np





def check_belong_to_curve(_x_data, _y_to_check, _curve_up_data_y, _curve_lw_data_y):
    """
    True se punto rispetta lowerbound e upperbound, False altrimenti
    
    :param _x_data: float
        la coordinata x del punto
    :param _y_to_check: float
        il valore dell'attributo da controllare del punto scelto
    :param _curve_up_data_y: numpy array, shape(0,5)
        i punti di upperbound dell'attributo da rispettare
    :param _curve_lw_data_y: numpy array, shape(0,5)
        i punti di lowerbound dell'attributo da rispettare
    
    :return: bool
        True se rispetta upperbound e lowerbound. False altrimenti
    """
    load_ratio = [0,25,50,75,100]
    value_interp_on_curve_up = np.interp(_x_data, load_ratio, _curve_up_data_y)
    value_interp_on_curve_lw = np.interp(_x_data, load_ratio, _curve_lw_data_y)
    if _y_to_check > value_interp_on_curve_up:
        # non rispetta upperbound
        return False
    if _y_to_check < value_interp_on_curve_lw:
        # non rispetta lowerbound
        return False
    return True


# Grafici 2d partizionati per analisi attributo di un Dataframe. Possibilità di salvare i subset del Dataframe 




import matplotlib.pyplot as plt





def plot_2d_rint(_df,_column_interv, _column_x, _column_y, _title = 'example', _filename = ''):
    """
    Creazione di molteplici plot per ogni valore distinto di un attributo di un Dataframe.
    Analisi bidimensionale.
    
    :param _df: Dataframe
        Dataframe da analizzare
    :param _column_interv: str
        colonna del Dataframe su cui partizionare i plot. Per ogni suo valore univoco verrà creato un plot e un subset.
    :param _column_x: str
        colonna del Dataframe da analizzare lungo a x
    :param _column_y: str
        colonna del Dataframe da analizzare lungo a y
    :param _title: str
        nome del grafico
    :param _filename: str
        percorso o nome del file png da salvare  
    """
    for value in _df[_column_interv].value_counts().index.tolist():
        df_sub = _df[_df[_column_interv] == value]
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot()
        ax.scatter(df_sub[_column_x], df_sub[_column_y], c='green', alpha=0.2, marker="s")
        plt.title('Plot {}, {} con valore {}'.format(_title,_column_interv, str(value)))
        plt.xlabel(_column_x)
        plt.ylabel(_column_y)
        plt.grid()
        plt.savefig(_filename + '_{}_{}.png'.format(_column_interv, str(value)))





def plot_combo_2d_rint(_df,_column_interv_1, _column_interv_2, _column_x, _column_y, _title = 'example', _filename = ''):
    """
    Creazione di molteplici plot per ogni valore distinto di due attributi(combinazione) di un Dataframe.
    Analisi bidimensionale. Salvataggio subset Dataframe.
    
    :param _df: Dataframe
        Dataframe da analizzare
    :param _column_interv_1: str
        prima colonna del Dataframe su cui partizionare i plot. Per ogni suo valore univoco verrà creato un plot e un subset.
    :param _column_interv_2: str
        seconda colonna del Dataframe su cui partizionare i plot. Per ogni suo valore univoco verrà creato un plot e un subset.
    :param _column_x: str
        colonna del Dataframe da analizzare lungo a x
    :param _column_y: str
        colonna del Dataframe da analizzare lungo a y
    :param _title: str
        nome del grafico
    :param _filename: str
        percorso o nome del file png da salvare  
    """
    for col_1 in _df[_column_interv_1].value_counts().index.tolist():
        for col_2 in _df[_column_interv_2].value_counts().index.tolist():
            title_plot = '{},{}={}_{}={}'.format(_title,_column_interv_1, str(col_1),_column_interv_2, str(col_2))
            df_sub = _df[(_df[_column_interv_1] == col_1) & (_df[_column_interv_2] == col_2)]
            if df_sub.shape[0] == 0:
                continue
            df_sub.to_csv(_filename + title_plot+'.csv')
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot()
            ax.scatter(df_sub[_column_x], df_sub[_column_y], c='green', alpha=0.2, marker="s")
            plt.title(title_plot)
            plt.xlabel(_column_x)
            plt.ylabel(_column_y)
            plt.grid()
            plt.savefig(_filename + title_plot +'.png')


# Normalizzazione 0-1




def normalize_min_max(_series, _min, _max):
    """
    Normalizzazione 0-1 di una features entro un minimo e massimo.
    
    :param _series: Series
        colonna da normalizzare nei range 0-1
    :param _min: float
        valore minimo per la normalizzazione
    :param _max: float
        valore massimo per la normalizzazione
    
    :return: float
        valore normalizzato
    """
    return (_series - float(_min))/(float(_max) - float(_min))

