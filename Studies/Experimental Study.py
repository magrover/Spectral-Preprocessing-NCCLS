# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:26:59 2023

@author: scrouse6
"""


import numpy as np
import matplotlib as mpl
import pylab as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import sklearn
from sklearn.cross_decomposition import PLSRegression
from scipy.special import erfinv

from Creation_Functions_Study import *
from Removal_Functions_Study import *

Methods = ['No Preprocessing','PCA','SRACLS','CDAE','BSS ICA','BSS PCA', 'NCCLS']
target_list = [0,1,2]
n_targets = 3
n_nontargets = 4
seed = 19

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})
mpl.rcParams['figure.dpi'] = 300

color=['orangered','royalblue','goldenrod','limegreen','darkviolet','slategray','chocolate','turquoise','dodgerblue','deeppink','seagreen']
marker=['o','^','s','*','D','X']

RamanMin=850       #Min is 100
RamanMax=1450       #Max is 3200
IRmin=1020        #Min is 648.8736  
IRmax=1420        #Max is 2998.2

indexmaxram=RamanMax-100
indexminram=RamanMin-100
indexmaxir=np.round(632-((IRmin-648.8736)/3.7232)).astype(int)
indexminir=np.round(632-((IRmax-648.8736)/3.7232)).astype(int)-1


Species_all = ['Nitrate','Nitrite','Sulfate','Carbonate','Phosphate','Acetate','Oxalate','Water']

readname = 'Experimental Data'
file_order = ['X_train_ram','X_train_ir','X_test_ram','X_test_ir','X_ref_ram','X_ref_ir','X_ref_ram_nontarget',
              'X_ref_ir_nontarget','Y_train','Y_test','X_ref_ram_total','X_ref_ir_total','wavenumber_ram','wavenumber_ir']

npz_file = np.load(readname + '.npz', allow_pickle=True)
myVars = vars()
for i in range(len(npz_file)):
    myVars[file_order[i]] = npz_file[file_order[i]]

"""
Preprocessing
"""

## Feature Selection
threshold = 0.95

X_train_ram_orig = X_train_ram.copy(); X_test_ram_orig = X_test_ram.copy(); X_ref_ram_orig = X_ref_ram.copy()
X_train_ir_orig = X_train_ir.copy(); X_test_ir_orig = X_test_ir.copy(); X_ref_ir_orig = X_ref_ir.copy()

wavenumber_ram, X_train_ram, X_test_ram, X_ref_ram_total = Cor_feature_selection(wavenumber_ram,X_train_ram,Y_train,X_test_ram,X_ref_ram_total,threshold)
wavenumber_ir, X_train_ir, X_test_ir, X_ref_ir_total = Cor_feature_selection(wavenumber_ir,X_train_ir,Y_train,X_test_ir,X_ref_ir_total,threshold)

X_ref_ram = X_ref_ram_total[:3,:].copy()
X_ref_ir = X_ref_ir_total[:3,:].copy()

"""
Plotting Residuals
"""
fontsize = 16
Error_limit = .50

X_error_residuals_ram = (Y_train@(X_ref_ram)-X_train_ram)
X_error_std_ram = np.std(X_error_residuals_ram)
X_error_mean_ram = np.mean(X_error_residuals_ram)
z_adjustment_ram = np.sqrt(2)*erfinv(2*(Error_limit)**(1/len(wavenumber_ram))-1)
sigma_estimate_ram = z_adjustment_ram*X_error_std_ram

X_error_residuals_ir = (Y_train@(X_ref_ir)-X_train_ir)
X_error_std_ir = np.std(X_error_residuals_ir)
X_error_mean_ir = np.mean(X_error_residuals_ir)
z_adjustment_ir = np.sqrt(2)*erfinv(2*(Error_limit)**(1/len(wavenumber_ir))-1)
sigma_estimate_ir = z_adjustment_ir*X_error_std_ir


"""
Residual Plotting with Individual Wavenumbers
"""

X_error_residuals_ram = (Y_train@(X_ref_ram)-X_train_ram)
X_error_std_ram = np.std(np.abs(X_error_residuals_ram), axis=0)
X_error_mean_ram = np.mean(X_error_residuals_ram, axis=0)
z_adjustment_ram = np.sqrt(2)*erfinv(2*(Error_limit)**(1/len(wavenumber_ram))-1)
sigma_estimate_ram = z_adjustment_ram*X_error_std_ram #+ X_error_mean_ram

X_error_residuals_ir = (Y_train@(X_ref_ir)-X_train_ir)
X_error_std_ir = np.std(np.abs(X_error_residuals_ir), axis=0)
X_error_mean_ir = np.mean(X_error_residuals_ir, axis=0)
z_adjustment_ir = np.sqrt(2)*erfinv(2*(Error_limit)**(1/len(wavenumber_ir))-1)
sigma_estimate_ir = z_adjustment_ir*X_error_std_ir #+ X_error_mean_ir

## Residual Scatter Plot
plt.figure()
plt.plot(wavenumber_ram, X_error_residuals_ram.T, marker='o', linestyle='None')
plt.plot(wavenumber_ram, wavenumber_ram*0 + sigma_estimate_ram + X_error_mean_ram, linestyle='--', color='k')
plt.plot(wavenumber_ram, wavenumber_ram*0 - sigma_estimate_ram + X_error_mean_ram, linestyle='--', color='k')
plt.xlabel('Raman Shift ($\mathrm{cm^{-1}}$)', fontsize=fontsize)
plt.ylabel('Intensity (Counts)', fontsize=fontsize)

plt.figure()
plt.plot(wavenumber_ir, X_error_residuals_ir.T, marker='o', linestyle='None')
plt.plot(wavenumber_ir, wavenumber_ir*0 + sigma_estimate_ir + X_error_mean_ir, linestyle='--', color='k')
plt.plot(wavenumber_ir, wavenumber_ir*0 - sigma_estimate_ir + X_error_mean_ir, linestyle='--', color='k')
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=fontsize)
plt.ylabel('Absorbance', fontsize=fontsize)
plt.xlim(np.max(wavenumber_ir),np.min(wavenumber_ir))

"""
Removal Functions
"""
spectra_all = [X_train_ram, X_test_ram, X_ref_ram, X_train_ir, X_test_ir, X_ref_ir]
spectra_names = ['Raman Training Data','Raman Testing Data','Raman Reference Data','IR Training Data','IR Testing Data','IR Reference Data']


X_pca_ram, _ = PCA_removal(X_train_ram, X_test_ram, n_species=n_targets)
X_autoencoder_ram, _ = Autoencoder_removal(X_train_ram, X_test_ram, max_trials = 1, epochs = 400, Simulated_examples = 0, seed = seed)
X_sracls_ram = SRACLS_removal(X_train_ram, X_ref_ram, X_test_ram, n_species=n_targets)
X_ccls_ram, y_ccls_ram, _ = Constrained_CLS_removal(X_train_ram, Y_train, X_test_ram, X_ref_ram)
X_bss_ram, Cor_ram, S0mcr_ram, S0ica_ram = BSS_removal(X_train_ram, X_test_ram, X_ref_ram, n_targets+n_nontargets, seed)
X_bsspca_ram = BSS_removal4(X_train_ram, X_test_ram, X_ref_ram, 0, n_targets+n_nontargets, seed)
method_list_ram = [X_test_ram, X_pca_ram, X_sracls_ram, X_autoencoder_ram, X_bss_ram, X_bsspca_ram, X_ccls_ram]

X_pca_ir, _ = PCA_removal(X_train_ir, X_test_ir, n_species=n_targets)
X_autoencoder_ir, _ = Autoencoder_removal(X_train_ir, X_test_ir, max_trials = 1, epochs = 400, Simulated_examples = 0, seed = seed)
X_sracls_ir = SRACLS_removal(X_train_ir, X_ref_ir, X_test_ir, n_species=n_targets)
X_ccls_ir, y_ccls_ir, _ = Constrained_CLS_removal(X_train_ir, Y_train, X_test_ir, X_ref_ir)
X_bss_ir, Cor_ir, S0mcr_ir, S0ica_ir = BSS_removal(X_train_ir, X_test_ir, X_ref_ir, n_targets+n_nontargets, seed)
X_bsspca_ir = BSS_removal4(X_train_ir, X_test_ir, X_ref_ir, 0, n_targets+n_nontargets, seed)
method_list_ir = [X_test_ir, X_pca_ir, X_sracls_ir, X_autoencoder_ir, X_bss_ir, X_bsspca_ir, X_ccls_ir]

"""
Real-time Removal
"""

X_autoencoder_ram_rt = np.zeros_like(X_autoencoder_ram)
X_pca_ram_rt = np.zeros_like(X_pca_ram); X_sracls_ram_rt = np.zeros_like(X_sracls_ram); X_ccls_ram_rt = np.zeros_like(X_ccls_ram)
X_bss_ram_rt = np.zeros_like(X_bss_ram); X_bsspca_ram_rt = np.zeros_like(X_bsspca_ram)

X_autoencoder_ir_rt = np.zeros_like(X_autoencoder_ir)
X_pca_ir_rt = np.zeros_like(X_pca_ir); X_sracls_ir_rt = np.zeros_like(X_sracls_ir); X_ccls_ir_rt = np.zeros_like(X_ccls_ir)
X_bss_ir_rt = np.zeros_like(X_bss_ir); X_bsspca_ir_rt = np.zeros_like(X_bsspca_ir)

X_autoencoder_ram_rt, _ = Autoencoder_removal_rt(X_train_ram, X_test_ram, max_trials = 100, epochs = 400, Simulated_examples = 0, seed = seed)
X_autoencoder_ir_rt, _ = Autoencoder_removal_rt(X_train_ir, X_test_ir, max_trials = 100, epochs = 400, Simulated_examples = 0, seed = seed)
for i in range(np.size(X_test_ram, axis=0)):
    X_pca_ram_rt[i:(i+1),:], _ = PCA_removal(X_train_ram, X_test_ram[i:(i+1),:], 1)
    X_sracls_ram_rt[i:(i+1),:] = SRACLS_removal(X_train_ram, X_ref_ram, X_test_ram[i:(i+1),:], 1)
    X_ccls_ram_rt[i:(i+1),:], C_pyomo_rt, _ = Constrained_CLS_removal(X_train_ram, Y_train, X_test_ram[i:(i+1),:], X_ref_ram)
    X_bss_ram_rt[i:(i+1),:], Cor_rt, S0mcr_rt, S0ica_rt = BSS_removal(X_train_ram, X_test_ram[i:(i+1),:], X_ref_ram, n_targets+n_nontargets, seed)
    X_bsspca_ram_rt[i:(i+1),:] = BSS_removal4(X_train_ram, X_test_ram[i:(i+1),:], X_ref_ram, 0, n_targets+1, seed)
    
    X_pca_ir_rt[i:(i+1),:], _ = PCA_removal(X_train_ir, X_test_ir[i:(i+1),:], 1)
    X_sracls_ir_rt[i:(i+1),:] = SRACLS_removal(X_train_ir, X_ref_ir, X_test_ir[i:(i+1),:], 1)
    X_ccls_ir_rt[i:(i+1),:], C_pyomo_rt, _ = Constrained_CLS_removal(X_train_ir, Y_train, X_test_ir[i:(i+1),:], X_ref_ir)
    X_bss_ir_rt[i:(i+1),:], Cor_rt, S0mcr_rt, S0ica_rt = BSS_removal(X_train_ir, X_test_ir[i:(i+1),:], X_ref_ir, n_targets+n_nontargets, seed)
    X_bsspca_ir_rt[i:(i+1),:] = BSS_removal4(X_train_ir, X_test_ir[i:(i+1),:], X_ref_ir, 0, n_targets+1, seed)
    
    print(i)
    
method_list_ram_rt = [X_test_ram, X_pca_ram_rt, X_sracls_ram_rt, X_autoencoder_ram_rt, X_bss_ram_rt, X_bsspca_ram_rt, X_ccls_ram_rt]
method_list_ir_rt = [X_test_ir, X_pca_ir_rt, X_sracls_ir_rt, X_autoencoder_ir_rt, X_bss_ir_rt, X_bsspca_ir_rt, X_ccls_ir_rt]

"""
Savitzky-Golay Filtering / Scaling
"""

filter_points = 7
filter_order = 2
filter_deriv = 1

method_list_ram_unfiltered = method_list_ram.copy()
method_list_ram_unfiltered_rt = method_list_ram_rt.copy()
method_list_ir_unfiltered = method_list_ir.copy()
method_list_ir_unfiltered_rt = method_list_ir_rt.copy()

X_train_ram = savgol_filter(X_train_ram.copy(), filter_points, filter_order, filter_deriv)
X_ref_ram = savgol_filter(X_ref_ram.copy(), filter_points, filter_order, filter_deriv)

for i in range(len(method_list_ram)):
    method_list_ram[i] = savgol_filter(method_list_ram[i].copy(), filter_points,filter_order,filter_deriv) 
    method_list_ram_rt[i] = savgol_filter(method_list_ram_rt[i].copy(), filter_points,filter_order,filter_deriv) 
    
X_train_ir = savgol_filter(X_train_ir.copy(), filter_points, filter_order, filter_deriv)
X_ref_ir = savgol_filter(X_ref_ir.copy(), filter_points, filter_order, filter_deriv)
for i in range(len(method_list_ir)):
    method_list_ir[i] = savgol_filter(method_list_ir[i].copy(), filter_points,filter_order,filter_deriv) 
    method_list_ir_rt[i] = savgol_filter(method_list_ir_rt[i].copy(), filter_points,filter_order,filter_deriv) 

"""
PLSR Quantification
"""

model_ram=PLSRegression(n_components=4, scale=True)
model_ram.fit(X_train_ram,Y_train)
y_hat_ram = np.zeros((np.hstack((np.shape(Y_test),len(Methods))))); y_hat_ram_rt = y_hat_ram.copy()
for i in range(len(Methods)):
    y_hat_ram[:,:,i] = model_ram.predict(method_list_ram[i])
    y_hat_ram_rt[:,:,i] = model_ram.predict(method_list_ram_rt[i])
    
model_ir=PLSRegression(n_components=4, scale=True)
model_ir.fit(X_train_ir,Y_train)
y_hat_ir = np.zeros((np.hstack((np.shape(Y_test),len(Methods))))); y_hat_ir_rt = y_hat_ir.copy()
for i in range(len(Methods)):
    y_hat_ir[:,:,i] = model_ir.predict(method_list_ir[i])
    y_hat_ir_rt[:,:,i] = model_ir.predict(method_list_ir_rt[i])

## Nonnegativity Output Constraints for all methods
y_hat_ram[y_hat_ram<0] = 0; y_hat_ram_rt[y_hat_ram_rt<0] = 0
y_hat_ir[y_hat_ir<0] = 0; y_hat_ir_rt[y_hat_ir_rt<0] = 0

"""
Error Metrics
"""

Score_R2_ram = np.zeros((len(Methods),np.size(Y_train, axis=1))); Score_R2_ir = Score_R2_ram.copy()
Score_RMSE_ram = Score_R2_ram.copy(); Score_RMSE_ir = Score_RMSE_ram.copy()
Score_RMSE_ram_rt = Score_RMSE_ram.copy(); Score_RMSE_ir_rt = Score_RMSE_ir.copy()
for i in range(len(Methods)):
    Score_R2_ram[i,:] = sklearn.metrics.r2_score(y_hat_ram[:,:,i], Y_test, multioutput = 'raw_values')
    Score_R2_ir[i,:] = sklearn.metrics.r2_score(y_hat_ir[:,:,i], Y_test, multioutput = 'raw_values')
    Score_RMSE_ram[i,:] = sklearn.metrics.mean_squared_error(y_hat_ram[:,:,i], Y_test, multioutput = 'raw_values', squared=False)
    Score_RMSE_ir[i,:] = sklearn.metrics.mean_squared_error(y_hat_ir[:,:,i], Y_test, multioutput = 'raw_values', squared=False)
    Score_RMSE_ram_rt[i,:] = sklearn.metrics.mean_squared_error(y_hat_ram_rt[:,:,i], Y_test, multioutput = 'raw_values', squared=False)
    Score_RMSE_ir_rt[i,:] = sklearn.metrics.mean_squared_error(y_hat_ir_rt[:,:,i], Y_test, multioutput = 'raw_values', squared=False)
    
## Adding a row for CCLS without PLSR
Score_R2_ram = np.vstack((Score_R2_ram, sklearn.metrics.r2_score(y_ccls_ram[:,:], Y_test, multioutput = 'raw_values')))
Score_R2_ir = np.vstack((Score_R2_ir, sklearn.metrics.r2_score(y_ccls_ir[:,:], Y_test, multioutput = 'raw_values')))
Score_RMSE_ram = np.vstack((Score_RMSE_ram, sklearn.metrics.mean_squared_error(y_ccls_ram[:,:], Y_test, multioutput = 'raw_values')))
Score_RMSE_ir = np.vstack((Score_RMSE_ir, sklearn.metrics.mean_squared_error(y_ccls_ir[:,:], Y_test, multioutput = 'raw_values')))

y_difference_ccls_ram = y_hat_ram[:,:,4] - Y_test
y_difference_ccls_ir = y_hat_ir[:,:,4] - Y_test

"""
Plotting
"""

title_lettering = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)']
linestyle = ['-','-','-','--','--','--','--']
alphas = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5]
s = 75
alpha = 0.7

## All References
handles = []; labels = []
fig, axs = plt.subplots(1, 2, figsize=(12,6))
axes = axs.ravel()
plt.subplots_adjust(wspace=.25,hspace=.5)
for i in range(2):
    ax = axes[i]
    ax.set_xlabel(r'Wavenumber (cm)$^{-1}$')
    if i == 0:
        for j in range(np.size(X_ref_ram_total, axis=0)):
            ax.plot(wavenumber_ram, X_ref_ram_total[j,:].T, color=color[j], linewidth=2, linestyle=linestyle[j], alpha=alphas[j])
            ax.set_title(title_lettering[i], loc='left', fontweight = 'bold')
            ax.set_ylabel(r'Counts')
            # ax.set_xlim((0,1.6))
            # ax.set_ylim((0,2.5))
    if i == 1:
        for j in range(np.size(X_ref_ir_total, axis=0)):
            ax.set_xlim(np.max(wavenumber_ir),np.min(wavenumber_ir))
            ax.plot(wavenumber_ir, X_ref_ir_total[j,:].T, color=color[j], linewidth=2, linestyle=linestyle[j], alpha=alphas[j])
            ax.set_title(title_lettering[i], loc='left', fontweight = 'bold')
            ax.set_ylabel(r'Absorbance')
        legend_elements1= [Line2D([0], [0], color=color[k], markeredgecolor='k', alpha=alpha, linestyle=linestyle[k], marker=marker[0], markersize=0, lw=2, label=Species_all[k]) for k in range(np.size(X_ref_ir_total, axis=0))]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (1.5, 1.0))
    
## Subplots Raman Quantification
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,10))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.5)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_targets):
            ax.scatter(Y_test[:,j],y_hat_ram[:,j,i], color=color[j], marker=marker[j], edgecolors='k', s=s, alpha=alpha)
            ax_min = np.max((np.min(Y_test[:,j]),np.min(y_hat_ram[:,j,i])))
            ax_max = np.min((np.max(Y_test[:,j]),np.max(y_hat_ram[:,j,i])))
            ax.plot([0, 2.5], [0, 2.5], color='k', zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Actual Conc. (M)')
        ax.set_ylabel(r'Predicted Conc. (M)')
        ax.set_xlim((0,1.6))
        ax.set_ylim((0,2.5))
        ax.text(0.05, 2.2, 'RMSE: ' + str(f'{np.mean(Score_RMSE_ram[i,:]):.3f}'))
    elif i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=8, lw=0, label='Nitrate'),
                            Line2D([0], [0], color=color[1], markeredgecolor='k', alpha=alpha, marker=marker[1], markersize=8, lw=0, label='Nitrite'),
                            Line2D([0], [0], color=color[2], markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=8, lw=0, label='Sulfate'),
                            Line2D([0], [0], color='k', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Parity Line')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.0))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots Raman RT Quantification
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,10))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.5)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_targets):
            ax.scatter(Y_test[:,j],y_hat_ram_rt[:,j,i], color=color[j], marker=marker[j], edgecolors='k', s=s, alpha=alpha)
            ax_min = np.max((np.min(Y_test[:,j]),np.min(y_hat_ram_rt[:,j,i])))
            ax_max = np.min((np.max(Y_test[:,j]),np.max(y_hat_ram_rt[:,j,i])))
            ax.plot([0, 2.5], [0, 2.5], color='k', zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Actual Conc. (M)')
        ax.set_ylabel(r'Predicted Conc. (M)')
        ax.set_xlim((0,1.6))
        ax.set_ylim((0,2.5))
        ax.text(0.05, 2.2, 'RMSE: ' + str(f'{np.mean(Score_RMSE_ram_rt[i,:]):.3f}'))
    elif i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=8, lw=0, label='Nitrate'),
                            Line2D([0], [0], color=color[1], markeredgecolor='k', alpha=alpha, marker=marker[1], markersize=8, lw=0, label='Nitrite'),
                            Line2D([0], [0], color=color[2], markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=8, lw=0, label='Sulfate'),
                            Line2D([0], [0], color='k', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Parity Line')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.0))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots Raman Spectra
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,8))
axes = axs.ravel()
plt.subplots_adjust(wspace=.6,hspace=.8)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        ax.plot(wavenumber_ram, X_test_ram[::4,:].T, color='k', linestyle='--', linewidth=2, alpha=alpha)
        if i >= 1:
            ax.plot(wavenumber_ram, method_list_ram_unfiltered[i][::4,:].T, color=color[0], linewidth=2, alpha=alpha)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Wavenumber (cm)$^{-1}$')
        ax.set_ylabel(r'Counts')
        ax.set_xlim((1000,1100))
    if i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=0, lw=2, label='Preprocessed'),
                            Line2D([0], [0], color='k', markeredgecolor='k', linestyle='--', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Original')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.4))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots Raman Spectra RT
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,8))
axes = axs.ravel()
plt.subplots_adjust(wspace=.6,hspace=.8)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        ax.plot(wavenumber_ram, X_test_ram[::4,:].T, color='k', linestyle='--', linewidth=2, alpha=alpha)
        if i >= 1:
            ax.plot(wavenumber_ram, method_list_ram_unfiltered_rt[i][::4,:].T, color=color[0], linewidth=2, alpha=alpha)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Wavenumber (cm)$^{-1}$')
        ax.set_ylabel(r'Counts')
        ax.set_xlim((1000,1100))
    if i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=0, lw=2, label='Preprocessed'),
                            Line2D([0], [0], color='k', markeredgecolor='k', linestyle='--', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Original')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.4))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots IR Quantification
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,10))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.5)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_targets):
            ax.scatter(Y_test[:,j],y_hat_ir[:,j,i], color=color[j], marker=marker[j], edgecolors='k', s=s, alpha=alpha)
            ax_min = np.max((np.min(Y_test[:,j]),np.min(y_hat_ir[:,j,i])))
            ax_max = np.min((np.max(Y_test[:,j]),np.max(y_hat_ir[:,j,i])))
            ax.plot([0, 2.5], [0, 2.5], color='k', zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Actual Conc. (M)')
        ax.set_ylabel(r'Predicted Conc. (M)')
        ax.set_xlim((0,1.6))
        ax.set_ylim((0,2.5))
        ax.text(0.05, 2.2, 'RMSE: ' + str(f'{np.mean(Score_RMSE_ir[i,:]):.3f}'))
    elif i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=8, lw=0, label='Nitrate'),
                            Line2D([0], [0], color=color[1], markeredgecolor='k', alpha=alpha, marker=marker[1], markersize=8, lw=0, label='Nitrite'),
                            Line2D([0], [0], color=color[2], markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=8, lw=0, label='Sulfate'),
                            Line2D([0], [0], color='k', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Parity Line')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.0))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots IR RT Quantification
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,10))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.5)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        for j in range(n_targets):
            ax.scatter(Y_test[:,j],y_hat_ir_rt[:,j,i], color=color[j], marker=marker[j], edgecolors='k', s=s, alpha=alpha)
            ax_min = np.max((np.min(Y_test[:,j]),np.min(y_hat_ir_rt[:,j,i])))
            ax_max = np.min((np.max(Y_test[:,j]),np.max(y_hat_ir_rt[:,j,i])))
            ax.plot([0, 2.5], [0, 2.5], color='k', zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Actual Conc. (M)')
        ax.set_ylabel(r'Predicted Conc. (M)')
        ax.set_xlim((0,1.6))
        ax.set_ylim((0,2.5))
        ax.text(0.05, 2.2, 'RMSE: ' + str(f'{np.mean(Score_RMSE_ir_rt[i,:]):.3f}'))
    elif i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=8, lw=0, label='Nitrate'),
                            Line2D([0], [0], color=color[1], markeredgecolor='k', alpha=alpha, marker=marker[1], markersize=8, lw=0, label='Nitrite'),
                            Line2D([0], [0], color=color[2], markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=8, lw=0, label='Sulfate'),
                            Line2D([0], [0], color='k', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Parity Line')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.9, 4.0))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots IR Spectra
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,8))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.8)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        ax.plot(wavenumber_ir, X_test_ir[::4,:].T, color='k', linestyle='--', linewidth=2, alpha=alpha)
        if i >= 1:
            ax.plot(wavenumber_ir, method_list_ir_unfiltered[i][::4,:].T, color=color[0], linewidth=2, alpha=alpha, zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Wavenumber (cm)$^{-1}$')
        ax.set_ylabel(r'Counts')
        ax.set_xlim(np.max(wavenumber_ir),np.min(wavenumber_ir))
    if i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=0, lw=2, label='Preprocessed'),
                            Line2D([0], [0], color='k', linestyle='--', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Original')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.7, 4.4))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")

## Subplots IR Spectra RT
handles = []; labels = []
fig, axs = plt.subplots(3, 3, figsize=(10,8))
axes = axs.ravel()
plt.subplots_adjust(wspace=.5,hspace=.8)
for i in range(len(Methods)+2):
    if i < 7:
        if i == 0:
            ax = axes[i]
        elif i > 0:
            ax = axes[i+2]
        ax.plot(wavenumber_ir, X_test_ir[::4,:].T, color='k', linestyle='--', linewidth=2, alpha=alpha)
        if i >= 1:
            ax.plot(wavenumber_ir, method_list_ir_unfiltered_rt[i][::4,:].T, color=color[0], linewidth=2, alpha=alpha, zorder=-1)
        ax.set_title(title_lettering[i] + ' ' + Methods[i], loc='left', fontweight = 'bold')
        ax.set_xlabel(r'Wavenumber (cm)$^{-1}$')
        ax.set_ylabel(r'Counts')
        ax.set_xlim(np.max(wavenumber_ir),np.min(wavenumber_ir))
    if i == 7:
        ax = axes[1]
        ax.axis("off")
        legend_elements1= [Line2D([0], [0], color=color[0], markeredgecolor='k', alpha=alpha, marker=marker[0], markersize=0, lw=2, label='Preprocessed'),
                            Line2D([0], [0], color='k', linestyle='--', markeredgecolor='k', alpha=alpha, marker=marker[2], markersize=0, lw=2, label='Original')]
        plt.legend(handles=legend_elements1, frameon=False, bbox_to_anchor = (-0.7, 4.4))
    elif i > 7:
        ax = axes[2]
        ax.axis("off")
