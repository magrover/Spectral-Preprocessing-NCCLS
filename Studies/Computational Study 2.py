# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:05:11 2023

@author: scrouse6
"""

import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression

from Removal_Functions_Study import *
from Creation_Functions_Study import *

savename = '4-15-24'

"""
Setup
"""

Methods = ['No Preprocessing','PCA','SRACLS','Autoencoder','BSS ICA','BSS PCA','NCCLS']
n_method = len(Methods)


"""
Iteration Setup
"""
std_target = np.array([10,20,15])
I_target = np.array([10000,10000,10000])
std_nontarget = np.array([20])
I_nontarget = np.array([10000]) 
wavenumber=np.arange((1000))
n_targets = np.size(I_target)
mean_target = np.array([300,600,650])

n_exper = 8
n_exper_test = 8
n_overlap = 8
n_noise = 8
n_replicates = 5

exper = np.logspace(np.log10(n_targets+1), 4, n_exper).round(0).astype(int)
exper_test = np.logspace(np.log10(1),4, n_exper_test).round(0).astype(int)
overlap = np.linspace(530, 600, n_overlap)
noise = np.logspace(0, log10(400), n_noise)

experiments = [exper, exper_test, overlap, noise]

exper_base = 35
exper_test_base = 15
overlap_base = 560
noise_base = 20

R2_total_exper = np.zeros((n_exper, n_replicates, n_method, n_targets))
RMSE_total_exper = np.zeros((n_exper, n_replicates, n_method, n_targets))
MAE_total_exper = np.zeros((n_exper, n_replicates, n_method, n_targets))
Time_total_exper = np.zeros((n_exper, n_replicates, 11))

R2_total_exper_test = np.zeros((n_exper, n_replicates, n_method, n_targets))
RMSE_total_exper_test = np.zeros((n_exper, n_replicates, n_method, n_targets))
MAE_total_exper_test = np.zeros((n_exper, n_replicates, n_method, n_targets))
Time_total_exper_test = np.zeros((n_exper, n_replicates, 11))

R2_total_overlap = np.zeros((n_overlap, n_replicates, n_method, n_targets))
RMSE_total_overlap = np.zeros((n_overlap, n_replicates, n_method, n_targets))
MAE_total_overlap = np.zeros((n_overlap, n_replicates, n_method, n_targets))
Time_total_overlap = np.zeros((n_overlap, n_replicates, 11))

R2_total_noise = np.zeros((n_noise, n_replicates, n_method, n_targets))
RMSE_total_noise = np.zeros((n_noise, n_replicates, n_method, n_targets))
MAE_total_noise = np.zeros((n_noise, n_replicates, n_method, n_targets))
Time_total_noise = np.zeros((n_noise, n_replicates, 11))


"""
Loop
"""
count = 0
for e in range(4):
    for i in range(len(experiments[e])):
        exper_temp = exper_base
        exper_test_temp = exper_test_base
        overlap_temp = overlap_base
        noise_temp = noise_base
        if e==0: 
            exper_temp = exper[i]
        if e==1:
            exper_test_temp = exper_test[i]
        if e==2:
            overlap_temp = overlap[i]
        if e==3: 
            noise_temp = noise[i]
        mean_nontarget = [overlap_temp]
        sigma = noise_temp
        

        for r in range(n_replicates):
            try:
                    
                X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget = Data_creation2(
                    wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma, n_train = exper_temp, n_test = exper_test_temp, seed = 19+r)
    
                """
                Feature Selection
                """
                
                SNR = 10
                wavenumber, X_train, X_test, X_ref = SNR_feature_selection(wavenumber,X_train,X_test,X_ref,sigma,sigma*SNR)
                
                """
                Removal Functions
                """
                t = np.zeros((12))
                t[0] = time.time()
                try:
                    X_pca,t[1] = PCA_removal(X_train, X_test, n_targets)
                except:
                    t[1]=t[0].copy()
                    X_pca = X_test.copy()
                t[2] = time.time()
                
                try:
                    X_autoencoder, t[3] = Autoencoder_removal(X_train, X_test, max_trials = 1, epochs = 60, Simulated_examples = 0, seed = 19+r)
                except:
                    t[3] = t[2].copy()
                    X_autoencoder = X_test.copy()
                t[4] = time.time()
                
                t[5] = t[4] ## No training for SRACLS
                try:
                    X_sracls = SRACLS_removal(X_train, X_ref, X_test, n_targets)
                except:
                    X_sracls = X_test.copy()
                t[6] = time.time()
                
                try:
                    X_ccls, C_pyomo, t[7] = Constrained_CLS_removal(X_train, y_train, X_test, X_ref)
                except:
                    t[7] = t[6].copy()
                    X_ccls = X_test.copy()
                t[8] = time.time()
                
                t[9] = t[8] ## No training for BSS
                try:
                    X_bss, Cor, S0mcr, S0ica = BSS_removal(X_train, X_test, X_ref, len(mean_target)+len(mean_nontarget), 19+r)
                except:
                    X_bss = X_test.copy()
                t[10] = time.time()
                
                try:
                    X_bss_sc = BSS_removal4(X_train, X_test, X_ref, 0, len(mean_target)+len(mean_nontarget), 19+r)
                except:
                    X_bss_sc = X_test.copy()
                t[11] = time.time()

                
                for m in range(1,12):
                    if e == 0:
                        Time_total_exper[i,r,m-1] = t[m]-t[m-1]
                        
                    if e == 1: 
                        Time_total_exper_test[i,r,m-1] = t[m]-t[m-1]
                        
                    if e == 2:
                        Time_total_overlap[i,r,m-1] = t[m]-t[m-1]
                        
                    if e == 3:
                        Time_total_noise[i,r,m-1] = t[m]-t[m-1]
                    
                """
                Quantification
                """
                
                method_list = [X_test, X_pca, X_sracls, X_autoencoder, X_bss, X_bss_sc, X_ccls]
                model=PLSRegression(n_components=len(mean_target), scale=True)
                model.fit(X_train,y_train)
                y_hat = np.zeros((np.hstack((np.shape(y_test),len(Methods)))))
                for m in range(len(Methods)):
                    y_hat[:,:,m] = model.predict(method_list[m]); y_hat[y_hat<0] = 0
                    if e == 0:
                        R2_total_exper[i,r,m,:] = r2_score(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                        RMSE_total_exper[i,r,m,:] = mean_squared_error(y_test, y_hat[:,:,m], multioutput = 'raw_values', squared = False)
                        MAE_total_exper[i,r,m,:] = mean_absolute_error(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                    
                    if e == 1:
                         R2_total_exper_test[i,r,m,:] = r2_score(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                         RMSE_total_exper_test[i,r,m,:] = mean_squared_error(y_test, y_hat[:,:,m], multioutput = 'raw_values', squared = False)
                         MAE_total_exper_test[i,r,m,:] = mean_absolute_error(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                         
                    if e == 2:
                        R2_total_overlap[i,r,m,:] = r2_score(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                        RMSE_total_overlap[i,r,m,:] = mean_squared_error(y_test, y_hat[:,:,m], multioutput = 'raw_values', squared = False)
                        MAE_total_overlap[i,r,m,:] = mean_absolute_error(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                        
                    if e == 3:
                        R2_total_noise[i,r,m,:] = r2_score(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                        RMSE_total_noise[i,r,m,:] = mean_squared_error(y_test, y_hat[:,:,m], multioutput = 'raw_values', squared = False)
                        MAE_total_noise[i,r,m,:] = mean_absolute_error(y_test, y_hat[:,:,m], multioutput = 'raw_values')
                    
                count += 1
                print(str(count) + '/' + str((n_exper+n_exper_test+n_overlap+n_noise)*n_replicates) + ' Done')
            except:
                count += 1
                print(str(count) + '/' + str((n_exper+n_exper_test+n_overlap+n_noise)*n_replicates) + ' Done')
                pass
            
               
"""
Saving
"""
np.savez(savename, 
          Time_total_exper = Time_total_exper, R2_total_exper = R2_total_exper, RMSE_total_exper = RMSE_total_exper, MAE_total_exper = MAE_total_exper,
          Time_total_exper_test = Time_total_exper_test, R2_total_exper_test = R2_total_exper_test, RMSE_total_exper_test = RMSE_total_exper_test, MAE_total_exper_test = MAE_total_exper_test,
          Time_total_overlap = Time_total_overlap, R2_total_overlap = R2_total_overlap, RMSE_total_overlap = RMSE_total_overlap, MAE_total_overlap = MAE_total_overlap,
          Time_total_noise = Time_total_noise, R2_total_noise = R2_total_noise, RMSE_total_noise = RMSE_total_noise, MAE_total_noise = MAE_total_noise,
          )
