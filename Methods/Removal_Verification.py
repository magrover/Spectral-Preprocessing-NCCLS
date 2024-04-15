# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:05:11 2023

@author: scrouse6
"""

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import pylab as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression

from Creation_Functions import *
from Removal_Functions import *

"""
Plotting Parameters
"""

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})
mpl.rcParams['figure.dpi'] = 300

color=['orangered','royalblue','goldenrod','limegreen','darkviolet','slategray','chocolate','turquoise','dodgerblue','deeppink','seagreen']
marker=['o','^','s','*','D','X']
s = 75
alpha = 0.7
lw = 2
target_labels = ['Target 1','Target 2','Target 3']

"""
System Setup
"""

mean_target = np.array([300,600,650])
std_target = np.array([10,20,15])
I_target = np.array([1,1,1])
mean_nontarget = np.array([560])
std_nontarget = np.array([20])
I_nontarget = np.array([1]) 
sigma=20/10000
n=50
seed=19
n_species = np.size(mean_target)
wavenumber=np.arange((1000))

Methods = ['No Preprocessing','PCA','SRACLS','CDAE','BSS ICA','BSS PCA','NCCLS']

X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget = Data_creation(
    wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma, 35, 15, seed)

"""
Feature Selection
"""

plt.figure()
for i in range(3):
    plt.plot(wavenumber, X_ref[i,:].T, color=color[i], alpha=alpha, label=target_labels[i])
plt.plot(wavenumber, X_ref_nontarget.T, color='k', linestyle='--', label='Non-target')
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Intensity')
plt.legend(frameon=False, bbox_to_anchor = (1.0, 1.0))  
plt.title('Reference Spectra')     

plt.figure()
plt.plot(wavenumber, X_train.T, color=color[2], alpha=alpha)
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Counts')
plt.title('Training Data')

plt.figure()
plt.plot(wavenumber, X_test.T, color=color[3], alpha=alpha)
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Counts')
plt.title('Process Data')

wavenumber_orig = wavenumber.copy()
SNR = 10
wavenumber, X_train, X_test, X_ref = SNR_feature_selection(wavenumber,X_train,X_test,X_ref,sigma*SNR)
X_ref_nontarget = X_ref_nontarget[:,wavenumber]

"""
Removal Functions
"""

X_pca = PCA_removal(X_train, X_test, n_species)
X_autoencoder = Autoencoder_removal(X_train, X_test, epochs = 400)
X_sracls = SRACLS_removal(X_train, X_ref, X_test, n_species)
X_ccls = NCCLS(X_train, y_train, X_test, X_ref)
X_bss = BSS_ICA(X_train, X_test, X_ref, len(mean_target)+len(mean_nontarget))
X_bss_sc = BSS_PCA(X_train, X_test, X_ref, len(mean_target)+len(mean_nontarget))

method_list = [X_test, X_pca, X_sracls, X_autoencoder, X_bss, X_bss_sc, X_ccls]
 
"""
Quantification
"""

model=PLSRegression(n_components=len(mean_target), scale=True)
model.fit(X_train,y_train)
y_hat = np.zeros((np.hstack((np.shape(y_test),len(Methods)))))
R2_total = np.zeros((len(Methods), n_species)); RMSE_total = R2_total.copy()
for i in range(len(Methods)):
    y_hat[:,:,i] = model.predict(method_list[i]); y_hat[y_hat<0] = 0
    RMSE_total[i,:] = mean_squared_error(y_test, y_hat[:,:,i], multioutput = 'raw_values', squared = False)
    
"""
Plotting
"""

title_lettering = ['a)','b)','c)','d)','e)','f)','g)','h)']
## Quantification Subplots
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(12,12))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.5)
for i in range(len(Methods)+2):
    ax = axes[i]
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_species):
            ax.scatter(y_test[:,j],y_hat[:,j,i], color=color[j], marker=marker[j], edgecolors='k', s=s, alpha=alpha)
            ax_min = np.min((np.min(y_test[:,j]),np.min(y_hat[:,j,i])))
            ax_max = np.max((np.max(y_test[:,j]),np.max(y_hat[:,j,i])))
            ax.plot([0, 1.2], [0, 1.2], color='k', zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Actual Conc.')
        ax.set_ylabel(r'Predicted Conc.')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1.4))
    if i == 0: 
        ax.set_ylim((0, 1.05*np.max(y_hat[:,:,i])))
        ax.text(0.08, 4.4, 'RMSE: ' + str(f'{np.mean(RMSE_total[i,:]):.3f}'))
    elif i == 3:
        ax.set_ylim((0, 1.05*np.max(y_hat[:,:,i])))
        ax.text(0.08, 3.85, 'RMSE: ' + str(f'{np.mean(RMSE_total[i,:]):.3f}'))
    elif i < 7: 
        ax.text(0.08, 1.2, 'RMSE: ' + str(f'{np.mean(RMSE_total[i,:]):.3f}'))
    elif i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=8, lw=0, label='Target 1'),
                            Line2D([0], [0], color=color[1], markeredgecolor='k', alpha=alpha, marker=marker[1], markersize=8, lw=0, label='Target 2'),
                            Line2D([0], [0], color=color[2], markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=8, lw=0, label='Target 3'),
                            Line2D([0], [0], color='k', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Parity Line')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.0))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

"""
Masking
"""
mask_array = np.zeros((np.shape(wavenumber)))
method_list_mask = method_list.copy()
for j in range(len(wavenumber)-1):
    if (wavenumber[j+1]-wavenumber[j]>1):
        mask_array[j:(j+2)] = np.array([1,1])
for i in range(len(Methods)):
    for j in range(np.size(method_list[0], axis=0)):
        method_list_mask[i][j,:] = np.ma.masked_array(method_list[i][j,:], mask = mask_array)
wavenumber_mask = np.ma.masked_array(wavenumber, mask_array)

## True and Reconstructed Spectra
method_list_true = y_test@X_ref
mask_array = np.zeros((np.shape(wavenumber)))
method_list_mask = method_list.copy()
method_list_true_mask = method_list_true.copy()
for j in range(len(wavenumber)-1):
    if (wavenumber[j+1]-wavenumber[j]>1):
        mask_array[j:(j+2)] = np.array([1,1])
for i in range(len(Methods)):
    for j in range(np.size(method_list[0], axis=0)):
        method_list_mask[i][j,:] = np.ma.masked_array(method_list[i][j,:], mask = mask_array)
wavenumber_mask = np.ma.masked_array(wavenumber, mask_array)
for i in range(np.size(method_list[0], axis=0)):
    method_list_true_mask[i,:] = np.ma.masked_array(method_list_true[i,:], mask_array)
fig, axs = plt.subplots(3, 3, figsize=(12,8))
axes = axs.ravel()
plt.subplots_adjust(wspace=.4,hspace=.8)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_species):
            if i == 0:
                ax.plot(wavenumber_mask, method_list_mask[i][:5,:].T, linestyle=':', color=color[2], linewidth=2, alpha=0.9)
                ax.plot(wavenumber_mask, method_list_true[:5,:].T, linestyle='--', color=color[1], linewidth=2, alpha=alpha)
            else:
                ax.plot(wavenumber_mask, method_list_mask[i][:5,:].T, color=color[0], linewidth=2, alpha=alpha)
                ax.plot(wavenumber_mask, method_list_true[:5,:].T, linestyle='--', color=color[1], linewidth=2, alpha=alpha)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax.set_ylabel(r'Intensity')
        ax.set_ylim((0,np.max((1.1,np.max(method_list_mask[i][:5,:])))))
    if i == 7:
        ax = axes[1]
        ax.axis('off')
        legend_elements1= [Line2D([0], [0], color=color[2], alpha=0.9, markersize=8, linestyle = ':', lw=2, label='Raw Process Spectra'),
            Line2D([0], [0], color=color[1], alpha=alpha, markersize=8, linestyle = '--', lw=2, label='True Targets'),
                            Line2D([0], [0], color=color[0], alpha=alpha, markersize=8, lw=2, label='Preprocessed')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.4, 4.4))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")



