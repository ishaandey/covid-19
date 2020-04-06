#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:50:30 2020

@author: ishaandey
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import math
 

def clean_data_trackingproject(df, state='all'):
    dropCols = ['hash', 'dateChecked', 'fips']
    df = df.groupby(by=['state','date']).agg('last').reset_index().drop(dropCols,axis=1).sort_values(by=['state','date'])
    
    df.columns = [cap(i) for i in df.columns.values]
    
    df['Date'] = pd.to_datetime(df.Date, format='%Y%m%d').dt.date
    if state!='all':
        try:
                return df[df.State==state]
        except:
            print('Unable to subset for state {}'.format(state))
    else:
        return df 


def clean_data_hopkins(df, country='all'):
    if country!='all':
        dropCols = ['Province/State', 'Country/Region','Lat','Long']
        country = df[df['Country/Region'] == country].drop(dropCols,axis=1).T.reset_index()
        country.columns = ['Date','Confirmed']
        country['Country'] = 'US'
        
        cleaned_country = clean_cols(country)
        return cleaned_country

    else:
        dropCols = ['Lat','Long']   
        temp = df.drop(dropCols,axis=1).groupby(by='Country/Region').sum().reset_index()
        countries = pd.melt(temp, id_vars=['Country/Region'], var_name='date', value_name='cases')
        countries.columns = ['Country','Date','Confirmed']
        
        cleaned_countries = countries.groupby('Country').apply(clean_cols)
        return cleaned_countries

            
def clean_data_nyt(df, level='state'):
    if level == 'state':
        states = df.groupby(by=['state','date']).agg('last')[['cases']].reset_index()
        states.columns = ['State','Date','Confirmed']
        
        cleaned_states = states.groupby('State').apply(clean_cols)
        return cleaned_states
    
    else:
        local =  df.groupby(by=['county','state','date']).agg('last')[['cases']].reset_index()
        local.columns = ['County','State','Date','Confirmed']
        
        cleaned_local = local.groupby('State').apply(clean_cols)
        return cleaned_local            
            
            
def clean_cols(df, rates=False, smooth_days=3):
    try:
        df['Date'] = pd.to_datetime(df.Date).dt.date
        df['EpidemicStartDate'] = df.sort_values(by='Date').loc[df.Confirmed!=0].Date.iloc[0]
        df['DaysElapsed'] = (df.Date - df.EpidemicStartDate).dt.days + 1

    except:
        print('Unable to convert dates properly!')
    
    df['NewConfirmed'] = df.Confirmed.diff(periods=1)
    
    # Takes the natural log not log10 s
    df['DaysElapsed_Log'] = np.log(df.DaysElapsed)
    df['Confirmed_Log'] = np.log(df.Confirmed)
    df['NewConfirmed_Log'] = np.log(df.NewConfirmed)
    
    if rates:
        # Haha idk math
        df['GrowthRate'] = df.NewConfirmed/df.NewConfirmed.shift(1)
        df['Ratio'] = df.Confirmed/df.Confirmed.shift(1)

        # Haha irdk math
        df['GrowthRate_Smooth'] = df.GrowthRate.rolling(smooth_days).sum()/smooth_days
        df['Ratio_Smooth'] = df.Ratio.rolling(smooth_days).sum()/smooth_days
        
    return df


def cap(s):
    return s[:1].upper() + s[1:]
