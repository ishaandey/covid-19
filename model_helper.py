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

def get_r2(x, y):
    """
    Lightweight alternative of evaluating R2 of a series
    """
    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def predict_lowess(x, y_hat, days=3, extend=False, verbose=False):
    """
    Predict absolute case counts for the next d days from a lowess regression
    """
    f = interpolate.interp1d(x, y_hat, fill_value='extrapolate')
    if extend:
        x_range = np.append(x, np.arange(days) + x[-1] + 1)
        y_pred = np.append(y_hat, f(np.arange(days) + x[-1] + 1))
        return x_range, y_pred
    else:
        x_range = np.arange(1, 1+days) + x[-1]
        y_pred = f(np.arange(1, 1+days) + x[-1])
        return x_range, y_pred
    
    
def predict_growth(x, y, r, params, days=3):
    x_range = []
    y_pred =  []
    r_pred = []
    
    x_previous = x[-1]
    y_previous = y[-1]
    r_previous = r[-1] 

    d = 0
    while d < days:
        x_t = x_previous + 1
        y_t = y_previous*r_previous
        r_t = x_t*params[0]+params[1]
        
        x_range.append(x_t)
        y_pred.append(y_t)
        r_pred.append(r_t)
        
        x_previous = x_t
        y_previous = y_t
        r_previous = r_t

        d += 1
        
    return x_range, y_pred, r_pred