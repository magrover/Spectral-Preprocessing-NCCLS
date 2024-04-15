# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:18:10 2023

@author: scrouse6
"""

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.qmc import LatinHypercube


def gaussian_curves(wavenumber, intensities, means, stds):
    n=np.size(means)
    X_ref=np.zeros((n,len(wavenumber)))
    for i in range(n):
        X_ref[i,:]=intensities[i]*np.exp((-(wavenumber-means[i])**2)/(2*stds[i]**2))
    X=sum(X_ref[i,:] for i in range(n))
    return X, X_ref

def Data_creation(wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma, n):
    
    n_target=len(mean_target)
    n_nontarget=len(mean_nontarget)
    I=10000
    targets = len(mean_target)
    nontargets = len(mean_nontarget)
    seed=19

    """
    References
    """
    X_ref = np.zeros((targets,len(wavenumber)))
    for i in range(targets):
        X_null,X_ref=gaussian_curves(wavenumber, I*I_target, mean_target, std_target)
    for i in range(nontargets):
        X_null,X_ref_nontarget=gaussian_curves(wavenumber, I*I_nontarget, mean_nontarget, std_nontarget)
    X_ref += np.random.normal(scale=sigma, size=(n_target,len(wavenumber)))
    X_ref_nontarget += np.random.normal(scale=sigma, size=(n_nontarget,len(wavenumber)))
    
    
    """
    Latin Hypercube Sampling
    """
    sampler = LatinHypercube(n_target, centered=False, seed=seed)
    sample = sampler.random(n=n)
    
    Y = sample.copy()
    X = np.zeros((n,len(wavenumber)))
    for i in range(n):
        X[i,:],X_gaus_ref=gaussian_curves(wavenumber, I*I_target*Y[i,:], mean_target, std_target)
    
    X = X.copy()+np.random.normal(scale=sigma, size=(n,len(wavenumber)))     
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=19)
    
    
    """
    Creating Test Species
    """
    sampler_test = LatinHypercube(n_nontarget, centered=False, seed=seed)
    sample_test = sampler_test.random(n=np.size(X_test,axis=0))
    Y_test=sample_test.copy()
    X_test_new=X_test.copy()
    X_test_true = X_test.copy()
    for i in range(np.size(X_test,axis=0)):
        X_test_new[i,:],X_gaus_ref=gaussian_curves(wavenumber, I*I_nontarget*Y_test[i,:], mean_nontarget, std_nontarget)
    
    X_test+=X_test_new.copy()
    y_test_nontarget = Y_test.copy()
    return X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget


## Created to allow for enumeration of both training and testing species
def Data_creation2(wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma, n_train, n_test, seed):
    
    n_target=len(mean_target)
    n_nontarget=len(mean_nontarget)
    n_wavenumber = len(wavenumber)
    I=1
    targets = len(mean_target)
    nontargets = len(mean_nontarget)
    n = n_train + n_test
    """
    References
    """
    X_ref = np.zeros((targets,len(wavenumber)))
    for i in range(targets):
        X_null,X_ref=gaussian_curves(wavenumber, I*I_target, mean_target, std_target)
    for i in range(nontargets):
        X_null,X_ref_nontarget=gaussian_curves(wavenumber, I*I_nontarget, mean_nontarget, std_nontarget)
    np.random.seed(seed=seed)
    X_ref += np.random.normal(scale=sigma, size=(n_target,len(wavenumber)))
    X_ref_nontarget += np.random.normal(scale=sigma, size=(n_nontarget,len(wavenumber)))
    
    
    """
    Latin Hypercube Sampling, Training
    """
    sampler = LatinHypercube(n_target, centered=False, seed=seed)
    sample = sampler.random(n=n_train)
    
    Y = sample.copy()
    X = np.zeros((n_train,len(wavenumber)))
    for i in range(n_train):
        X[i,:],X_gaus_ref=gaussian_curves(wavenumber, I*I_target*Y[i,:], mean_target, std_target)
    
    np.random.seed(seed=seed+100)
    X_train = X.copy()+np.random.normal(scale=sigma, size=(n_train,len(wavenumber))) 
    y_train = Y.copy()    
    
    """
    Creating Test Species
    """
    sampler_test = LatinHypercube(n_target + n_nontarget, centered=False, seed=seed+200)
    sample_test = sampler_test.random(n=n_test)
    Y_test=sample_test.copy()
    X_test_target = np.zeros((n_test, n_wavenumber)); X_test_nontarget = np.zeros((n_test, n_wavenumber))
    
    for i in range(n_test):
        X_test_target[i,:],X_gaus_ref=gaussian_curves(wavenumber, I*I_target*Y_test[i,:targets], mean_target, std_target)
        X_test_nontarget[i,:],X_gaus_ref=gaussian_curves(wavenumber, I*I_nontarget*Y_test[i,targets:], mean_nontarget, std_nontarget)
    
    np.random.seed(seed=seed+300)
    X_test = X_test_target + X_test_nontarget + np.random.normal(scale=sigma, size=(n_test,len(wavenumber))) 
    y_test = Y_test[:,:targets].copy()
    y_test_nontarget = Y_test[:,targets:].copy()
    return X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget
