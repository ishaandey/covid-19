#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:16:51 2020

@author: ishaandey
"""

import pandas as pd
import numpy as np
import math

import statsmodels.api as sm
from sklearn.metrics import r2_score

from scipy.optimize import curve_fit
from scipy import interpolate


containment = pd.read_csv('containment.txt')
containment.columns = ['State','Date']

print('memes')
def annotate_containment(fig, lookup=containment, items=[], library='plotly'):
    global containment
    if library == 'plotly':
        for i in items:
            fig.add_shape(dict(type="line", 
                               x0=lookup[lookup['State']==i].Date.values[0], y0=1,
                               x1=lookup[lookup['State']==i].Date.values[0], y1=100000, 
                               line=dict(color="RoyalBlue",width=3)))
        fig.update_shapes(dict(xref='x', yref='y'))
        return fig
