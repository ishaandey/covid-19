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

def regress_lowess(x, y, verbose=True):
    y_pred = sm.nonparametric.lowess(y, x, frac=7/len(x), it=0, return_sorted=False)
    r2 = r2_score(y, y_pred)    
    
    if verbose:
        print('R2 for LOWESS: %2.5f'%r2)
        
    return y_pred

def predict(x, y, days=1, extend=False, verbose=False):
    f = interpolate.interp1d(x, y, fill_value='extrapolate')
    if extend:
        x_new = np.append(x, np.arange(days) + x[-1] + 1)
        y_new = np.append(y, f(np.arange(days) + x[-1] + 1))
        return (x_new, y_new)
    else:
        x_range = np.arange(1, 1+days) + x[-1]
        y_pred = f(np.arange(1, 1+days) + x[-1])
        return x_range, y_pred
