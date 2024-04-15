# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:40:44 2023

@author: scrouse6
"""

import numpy as np
import matplotlib as mpl
import pylab as plt


from Removal_Functions_Study import *
from Creation_Functions_Study import *


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})
mpl.rcParams['figure.dpi'] = 300

readname = '3-19-24'
file_order = ['Time_total_exper', 'Time_total_exper_test','Time_total_overlap', 'Time_total_noise', 
              'R2_total_exper', 'R2_total_exper_test','R2_total_overlap', 'R2_total_noise',
              'RMSE_total_exper', 'RMSE_total_exper_test', 'RMSE_total_overlap', 'RMSE_total_noise', 
              'MAE_total_exper', 'MAE_total_exper_test', 'MAE_total_overlap', 'MAE_total_noise']

npz_file = np.load(readname + '.npz', allow_pickle=True)
myVars = vars()
for i in range(len(npz_file)):
    myVars[file_order[i]] = npz_file[file_order[i]]

Methods = ['No Preprocessing','PCA','SRACLS','CDAE','BSS ICA','BSS PCA','NCCLS']
Methods_Time = ['PCA Train','PCA Predict','CDAE Train','CDAE Predict','SRACLS','CCLS Train','CCLS Predict','BSS_ICA','BSS_PCA']

color=['orangered','royalblue','goldenrod','limegreen','chocolate','slategray','darkviolet','turquoise','dodgerblue','deeppink','seagreen']
marker=['o','s','^','+','*','X','D','1']
color_time=['royalblue','royalblue','goldenrod','goldenrod','limegreen','darkviolet','darkviolet','slategray','chocolate']
marker_time=['^','v','s','h','*','D','d','X','+']

n_exper = 8
n_exper_test = 8
n_overlap = 8
n_noise = 8
n_replicates = 5
n_methods = 7
n_targets = 3
    
target = 1
time_points = [0,1,2,3,5,6,7,9,10]

Time_total = [Time_total_exper[:,:,time_points], Time_total_exper_test[:,:,time_points], Time_total_overlap[:,:,time_points], Time_total_noise[:,:,time_points]]
R2_total = [R2_total_exper, R2_total_exper_test, R2_total_overlap, R2_total_noise]
RMSE_total = [RMSE_total_exper, RMSE_total_exper_test, RMSE_total_overlap, RMSE_total_noise]
MAE_total = [MAE_total_exper, MAE_total_exper_test, MAE_total_overlap, MAE_total_noise]

## Find Averages
Time_avg = [np.mean(Time_total_exper, axis=1), np.mean(Time_total_exper_test, axis=1), np.mean(Time_total_overlap, axis=1), np.mean(Time_total_noise, axis=1)]
R2_avg = [np.mean(R2_total_exper, axis=1), np.mean(R2_total_exper_test, axis=1), np.mean(R2_total_overlap, axis=1), np.mean(R2_total_noise, axis=1)]
RMSE_avg = [np.mean(RMSE_total_exper, axis=1), np.mean(RMSE_total_exper_test, axis=1), np.mean(RMSE_total_overlap, axis=1), np.mean(RMSE_total_noise, axis=1)]
MAE_avg = [np.mean(MAE_total_exper, axis=1), np.mean(MAE_total_exper_test, axis=1), np.mean(MAE_total_overlap, axis=1), np.mean(MAE_total_noise, axis=1)]

## Find Minimums and Maximums
Time_min = [np.min(Time_total_exper, axis=1), np.min(Time_total_exper_test, axis=1), np.min(Time_total_overlap, axis=1), np.min(Time_total_noise, axis=1)]
R2_min = [np.min(R2_total_exper, axis=1), np.min(R2_total_exper_test, axis=1), np.min(R2_total_overlap, axis=1), np.min(R2_total_noise, axis=1)]
RMSE_min = [np.min(RMSE_total_exper, axis=1), np.min(RMSE_total_exper_test, axis=1), np.min(RMSE_total_overlap, axis=1), np.min(RMSE_total_noise, axis=1)]
MAE_min = [np.min(MAE_total_exper, axis=1), np.min(MAE_total_exper_test, axis=1), np.min(MAE_total_overlap, axis=1), np.min(MAE_total_noise, axis=1)]

Time_max = [np.max(Time_total_exper, axis=1), np.max(Time_total_exper_test, axis=1), np.max(Time_total_overlap, axis=1), np.max(Time_total_noise, axis=1)]
R2_max = [np.max(R2_total_exper, axis=1), np.max(R2_total_exper_test, axis=1), np.max(R2_total_overlap, axis=1), np.max(R2_total_noise, axis=1)]
RMSE_max = [np.max(RMSE_total_exper, axis=1), np.max(RMSE_total_exper_test, axis=1), np.max(RMSE_total_overlap, axis=1), np.max(RMSE_total_noise, axis=1)]
MAE_max = [np.max(MAE_total_exper, axis=1), np.max(MAE_total_exper_test, axis=1), np.max(MAE_total_overlap, axis=1), np.max(MAE_total_noise, axis=1)]

exper = np.logspace(np.log10(n_targets+1), 4, n_exper).round(0).astype(int)
exper_test = np.logspace(np.log10(1),4, n_exper_test).round(0).astype(int)
overlap = np.linspace(530, 600, n_overlap)
noise = np.logspace(0, log10(400), n_noise)/100 ##In noise percent

## Calculation of Overlap Percent
def overlap_calculation(overlap):
    std_target = np.array([10,20,15])
    I_target = np.array([1,1,1])
    std_nontarget = np.array([20])
    I_nontarget = np.array([1]) 
    wavenumber=np.arange((1000))
    n_targets = np.size(I_target)
    mean_target = np.array([300,600,650])
    overlap_percent = np.zeros((len(overlap)))
    X_ref_nontarget_overlap = np.zeros((len(overlap), len(wavenumber)))
    for i in range(len(overlap)):
        overlap_temp = overlap[i]
        mean_nontarget = [overlap_temp]
        X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget = Data_creation2(
          wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma=0, n_train = 20, n_test = 50, seed=19)
        X_ref_nontarget_overlap[i] = X_ref_nontarget.copy()
        overlap_area = np.trapz(np.min(np.vstack((X_ref[1:2,:], X_ref_nontarget)), axis=0))
        target_area = np.trapz(X_ref[1:2,:])
        overlap_percent[i] = (overlap_area / target_area)*100
    return overlap_percent

overlap_percent = overlap_calculation(overlap)
## Overlap for non-target at 560 is: 31.7411

Experiments = ['Number of Training Data', 'Number of Process Data', 'Nontarget Peak Overlap (%)','Noise (% of max peak height)']
Experiment_save = ['Train Data', 'Test Data', 'Overlap', 'Noise']
experiments = [exper, exper_test, overlap_percent, noise]
title_lettering = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)']

alpha_bar = 0.7
alpha_point = 1.0

"""
Plots
"""
 
## Time Reduced Plots
for j in range(len(Experiments)):
    handles = []; labels = []
    fig, axs = plt.subplots(1, 2, figsize=(12,6), sharex=True)
    axes = axs.ravel()
    # plt.suptitle(r'Computational Time: ' + Experiments[j])
    plt.subplots_adjust(wspace=.4,hspace=.25)
    for i in range(len(time_points)):
        if i == 2 or i == 3 or i == 7:
            ax = axes[0]
        # elif i == 2:
        #     ax = axes[2]
        else:
            ax = axes[1]
        ax.scatter(experiments[j], Time_avg[j][:,i], color=color_time[i], marker=marker_time[i], label=Methods_Time[i], s=75, edgecolors='k', alpha=alpha_point )
        ax.errorbar(experiments[j], (Time_max[j][:,i]+Time_min[j][:,i])/2, yerr=(Time_max[j][:,i]-Time_min[j][:,i])/2, fmt='none', ecolor=color_time[i], capsize=4, capthick=2, alpha=alpha_bar, zorder=-1)
        
    for s in range(2):
        ax = axes[s]
        ax.set_title(title_lettering[s], loc='left', fontweight = 'bold')
        ax.set_xlabel(Experiments[j])
        ax.set_ylabel(r'Time (s)')
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base = 10.0, subs=(1,), numticks=12))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,subs='all',numticks=12))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        handles_ , labels_ = ax.get_legend_handles_labels()
        handles = handles + handles_; labels = labels + labels_
        if s == 1:
            ax.legend(handles, labels, bbox_to_anchor = (1.9, 1.02), fontsize = 16)
        if j == 2:
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.set_xlim((1,110))
            

subregion = [[3, 11000, 0, 0.25],
             [0.8, 11000, 0, 0.25],
             [7.5, 105, 0, 0.25],
             [0.008, 5, 0, 0.25]]

legend_loc = [[0.1, 1.1],
              [0.1, 1.1],
              [0.1, 1.1],
              [0.1, 1.1]]

## RMSE Inset Plots
for j in range(len(Experiments)):
    handles = []; labels = []
    fig, ax = plt.subplots(figsize=(8,4), sharex=True)
    for i in range(len(Methods)):
        ax.scatter(experiments[j], RMSE_avg[j][:,i,target], color=color[i], marker=marker[i], label=Methods[i], alpha=alpha_point, s=75, edgecolors='k', )
        ax.errorbar(experiments[j], (RMSE_max[j][:,i,target]+RMSE_min[j][:,i,target])/2, yerr=(RMSE_max[j][:,i,target]-RMSE_min[j][:,i,target])/2, fmt='none', ecolor=color[i], capsize=4, alpha=alpha_bar, capthick=2, zorder=-1)
    
    ax.set_xlabel(Experiments[j])
    ax.set_ylabel(r'RMSE (Target 2)')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base = 10.0, subs=(1,), numticks=12))
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,subs='all',numticks=12))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim((0,1.05*np.max((RMSE_max[j][:,:,target]))))
    handles_ , labels_ = ax.get_legend_handles_labels()
    handles = handles + handles_; labels = labels + labels_
    ax.legend(handles, labels, bbox_to_anchor = legend_loc[j], fontsize = 16)
    if j >= 2:
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        
    x1, x2, y1, y2 = subregion[j]  # subregion of the original image
    axins = ax.inset_axes(
        [1.12, 0.84, 1, 1], #[x0, y0, width, height]
        xlim=(x1, x2), ylim=(y1, y2))
    for i in range(len(Methods)):
        axins.scatter(experiments[j], RMSE_avg[j][:,i,target], color=color[i], marker=marker[i], label=Methods[i], alpha=alpha_point, s=75, edgecolors='k', )
        axins.errorbar(experiments[j], (RMSE_max[j][:,i,target]+RMSE_min[j][:,i,target])/2, yerr=(RMSE_max[j][:,i,target]-RMSE_min[j][:,i,target])/2, fmt='none', ecolor=color[i], capsize=4, alpha=alpha_bar, capthick=2, zorder=-1)
    axins.set_xlabel(Experiments[j])
    axins.set_ylabel(r'RMSE (Target 2)')
    axins.set_xscale('log')
    axins.xaxis.set_major_locator(mpl.ticker.LogLocator(base = 10.0, subs=(1,), numticks=12))
    axins.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,subs='all',numticks=12))
    axins.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if j >= 2:
        axins.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.indicate_inset_zoom(axins, edgecolor="black")

## NCCLS average (one process spectrum): 0.0115903
## PCA average (one process spectrum): 0.0921069

"""
Explanation of Runs
"""
plt.figure(figsize=(10,12))
plt.subplots_adjust(wspace=.25,hspace=0)

"""
Overlap
"""

## Different amounts of overlap
std_target = np.array([10,20,15])
I_target = np.array([1,1,1])
std_nontarget = np.array([20])
I_nontarget = np.array([1]) 
wavenumber=np.arange((1000))
n_targets = np.size(I_target)
mean_target = np.array([300,600,650])

overlap_percent = np.zeros((len(overlap)))
X_ref_nontarget_overlap = np.zeros((len(overlap), len(wavenumber)))

for i in range(len(overlap)):
    overlap_temp = overlap[i]
    mean_nontarget = [overlap_temp]

    X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget = Data_creation2(
      wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma=0, n_train = 20, n_test = 50, seed=19)
    
    X_ref_nontarget_overlap[i] = X_ref_nontarget.copy()
    overlap_area = np.trapz(np.min(np.vstack((X_ref[1:2,:], X_ref_nontarget)), axis=0))
    target_area = np.trapz(X_ref[1:2,:])
    overlap_percent[i] = (overlap_area / target_area)*100


ax = plt.subplot(2,2,1)
ax.set_xlim((200,800))
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylabel('Normalized Stacked Spectra References')
ax.set_title(title_lettering[0], loc='left', fontweight = 'bold')
ax.text(265,10.45, 'Non-target Overlap', fontsize=24)
for i in range(len(overlap)):
    for j in range(3):
        ax.plot(wavenumber, X_ref[j,:].T + i*1.25, color=color[j], linewidth = 1)
    ax.plot(wavenumber, X_ref_nontarget_overlap[i,:].T + i*1.25, color='k', linestyle = '--', linewidth = 1)
    ax.plot(wavenumber, np.ones((np.shape(wavenumber)))*i*1.25, color='k', linewidth = 1)
    ax.text(overlap[0]-150, i*1.25 + 0.40 , str(overlap_percent[i].round(1)) + ' %', fontsize = 18)
    
    
"""
Noise
"""

std_target = np.array([10,20,15])
I_target = np.array([1,1,1])
std_nontarget = np.array([20])
I_nontarget = np.array([1]) 
wavenumber=np.arange((1000))
n_targets = np.size(I_target)
mean_target = np.array([300,600,650])
mean_nontarget = np.array([560])

noise_percent = np.zeros((len(noise)))
X_ref_nontarget_noise = np.zeros((len(noise), len(wavenumber)))
X_ref_noise = []

for i in range(len(noise)):
    noise_temp = noise[i]/100
    sigma = noise_temp

    X_train, y_train, X_test, y_test, X_ref, X_ref_nontarget, y_test_nontarget = Data_creation2(
      wavenumber, mean_target, std_target, I_target, mean_nontarget, std_nontarget, I_nontarget, sigma=sigma, n_train = 20, n_test = 50, seed=19)
    
    X_ref_nontarget_noise[i] = X_ref_nontarget.copy()
    X_ref_noise = X_ref_noise + [X_ref.copy()]

ax = plt.subplot(2,2,2)
ax.set_xlim((200,800))
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_title(title_lettering[1], loc='left', fontweight = 'bold')
ax.text(265,10.45, 'Measurement Noise', fontsize=24)
for i in range(len(overlap)):
    for j in range(3):
        ax.plot(wavenumber, X_ref_noise[i][j,:].T + i*1.25, color=color[j], linewidth = 1)
    ax.plot(wavenumber, X_ref_nontarget_noise[i,:].T + i*1.25, color='k', linestyle = '--', linewidth = 1)
    ax.plot(wavenumber, np.ones((np.shape(wavenumber)))*i*1.25, color='k', linewidth = 1)
    ax.text(overlap[0]-150, i*1.25 + 0.40 , str(f'{noise[i]:.3f}') + ' %', fontsize = 18)

"""
Table of Experiments
"""


rows = ['']
columns = ['Exp','1','2','3','4','5','6','7','8']
colors = plt.cm.BuPu(np.linspace(0, 0.4, len(columns)))
ax = plt.subplot(2,1,2)
ax.text(-.055, 0.035, title_lettering[2], fontsize=24, fontweight = 'bold')
ax.text(-.050,0.035, 'Training Data', fontsize=24)
exper_blank = np.hstack((np.array(['Data']).reshape(1,-1), exper.reshape(1,-1).astype(str)))
the_table = ax.table(cellText=exper_blank, rowLabels=rows, colColours=colors, colLabels=columns, bbox=[0,0.5,1,0.25])
the_table.set_fontsize(30)
ax.axis('off')
table_props = the_table.properties()
table_cells = table_props['children']
for cell in table_cells: cell.set_height(0.2)
the_table.scale(1,1.1)
ax.axis('tight')

## Testing Experiments
rows = ['']
columns = ['Exp','1','2','3','4','5','6','7','8']
colors = plt.cm.RdPu(np.linspace(0, 0.4, len(columns)))
ax = plt.subplot(2,1,2)
ax.text(-.055, -0.012, title_lettering[3], fontsize=24, fontweight = 'bold')
ax.text(-.050, -0.012, 'Process Data', fontsize=24)
exper_blank = np.hstack((np.array(['Data']).reshape(1,-1), exper_test.reshape(1,-1).astype(str)))
the_table = ax.table(cellText=exper_blank, rowLabels=rows, colColours=colors, colLabels=columns, bbox=[0,0.1,1,0.25])
the_table.set_fontsize(30)
ax.axis('off')
table_props = the_table.properties()
table_cells = table_props['children']
for cell in table_cells: cell.set_height(0.2)
the_table.scale(1,1.1)
ax.axis('tight')





