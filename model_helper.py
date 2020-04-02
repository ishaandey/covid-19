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


# S/o to aatishb for framework
def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))

def exponential(t, a, b, c):
    return a * np.exp(b * t) + c

def linear(t, a, c):
    return a*t + c

def get_r2(popt, how, x, y):
    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def regress(x, y, how='exponential', verbose=True):
    try:
        if how.lower() == 'exponential':
            popt, pcov = curve_fit(exponential, x, y, maxfev=10000, bounds=([0,0,-100],[100,0.9,100]))
            y_pred = exponential(x, *popt)
        elif how.lower() == 'linear':
            popt, pcov = curve_fit(linear, x, y, maxfev=10000)  
            y_pred = linear(x, *popt)
        elif how.lower() == 'logistic':   
            popt, pcov = curve_fit(logistic, x, y, maxfev=10000)   
            y_pred = logistic(x, *popt)
        elif how.lower() == 'lowess':
            y_pred = sm.nonparametric.lowess(y, x, frac=7/len(x), it=0, return_sorted=False)
            popt = None
        else:
            print('regress(how=) currently accepts exponential, linear, logistic, and lowess!')
            return None
    except RuntimeError:
        print('Unable to fit {} curve to data'.format(how))
        return None

    r2 = r2_score(y, y_pred)    
    
    if verbose:
        print('R2 for {h}: %2.5f'.format(h=how)%r2)
        if popt is not None:
            print('Model Params: {}'.format(popt))
        
    return (r2, y_pred, popt)

def predict(x, y, days=1, extend=False, verbose=False):
    f = interpolate.interp1d(x, y, fill_value='extrapolate')
    if extend:
        x_new = np.append(x, np.arange(1, 1+days) + x[-1])
        y_new = np.append(y, f(np.arange(1, 1+days) + x[-1]))
        return (x_new, y_new)
    else:
        x_range = np.arange(1, 1+days) + x[-1]
        y_pred = f(np.arange(1, 1+days) + x[-1])
        return x_range, y_pred
