# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:01:55 2023

@author: scrouse6
"""

import numpy as np
import os
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from sklearn.decomposition import FastICA, PCA 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from pyomo.environ import *
import scipy
import time

# import keras_tuner
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.optimizers import Adadelta



def PCA_removal(X_train, X_test, n_species):
    ## Scikit-Learn Implementation
    seed = 19
    pca_transform = PCA(n_components=n_species, random_state=seed)
    X_train_transformed = pca_transform.fit_transform(X_train); X_train_pca = pca_transform.inverse_transform(X_train_transformed)
    t = time.time()
    X_test_transformed = pca_transform.transform(X_test); X_test_pca = pca_transform.inverse_transform(X_test_transformed)
    X_removed = X_test_pca
    
    ## Matrix Implementation
    # n = np.size(X_train, axis=0)
    # X_train_avg = np.mean(X_train, axis=0).reshape(1,-1)
    # Phi = (1/(n-1))*(X_train.T - X_train_avg.T)@(X_train.T - X_train_avg.T).T
    # eigval, eigvec = np.linalg.eig(Phi)
    # eigsort = np.argsort(eigval)
    # eigval = eigval[eigsort]; eigvec[:,eigsort]
    # X_removed = np.zeros(np.shape(X_test))
    # for i in range(np.size(X_test, axis=0)):
    #     eig_sum = np.zeros(np.size(X_test, axis=1))
    #     for j in range(n_species):
    #         print(np.shape(X_test), np.shape(X_train_avg), np.shape(eigvec))
    #         eig_sum = eig_sum + (X_test[i:(i+1),:]@eigvec[:,j:(j+1)] - X_train_avg@eigvec[:,j:(j+1)])@eigvec[:,j:(j+1)].T
    #     X_removed[i,:] = X_train_avg + eig_sum
    
    return X_removed, t

def SRACLS_removal(X_train, X_ref, X_test, n_species):
    seed = 19
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref

    pca = PCA(n_components=n_species, random_state = seed)
    pca.fit(E)
    P_pca = pca.components_
    X_ref_pca = np.vstack((X_ref, P_pca))

    Y_pca_resid = X_test@X_ref_pca.T@np.linalg.inv(X_ref_pca@X_ref_pca.T)
    X_removed = Y_pca_resid[:,:np.size(X_ref, axis=0)]@X_ref_pca[:np.size(X_ref, axis=0),:] 
    return X_removed

def Autoencoder_removal(X_train, X_test, max_trials = 32, epochs = 50, Simulated_examples = 0, seed=19):
    # Extends domain to be divisible by 2^3 for convolutional autoencoder
    if np.size(X_train, axis=1) % 8 != 0:
        mod = 7 - (np.size(X_train, axis=1) % 8)
        X_train = np.hstack((X_train, X_train[:,-2:-1]*np.ones((np.size(X_train, axis=0), (mod+1)))))
        X_test = np.hstack((X_test, X_test[:,-2:-1]*np.ones((np.size(X_test, axis=0), (mod+1))))) 
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ## Hyperparameter Tuning
    # def build_model(hp):
    #     model = keras.Sequential()
        
        
    #     model.add(Conv1D(filters=hp.Int('filters1', min_value=16, max_value=124, step=16), kernel_size=hp.Int('kernel1', min_value=3, max_value=21, step=2), padding='same',activation='relu', input_shape = (1, np.size(X_train,axis=1))))
    #     model.add(ReLU())
    #     model.add(Dropout(rate=hp.Float("dropout", min_value=0, max_value=0.6, sampling="linear")))
    #     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        
    #     model.add(Conv1D(filters=hp.Int('filters2', min_value=8, max_value=64, step=8), kernel_size=hp.Int('kernel2', min_value=3, max_value=21, step=2), padding='same',activation='relu'))
    #     model.add(ReLU())
    #     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        
    #     model.add(Conv1D(filters=hp.Int('filters3', min_value=8, max_value=64, step=8), kernel_size=hp.Int('kernel3', min_value=3, max_value=21, step=2), padding='same',activation='relu'))
    #     model.add(ReLU())
    #     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        
        
    #     model.add(UpSampling1D(size=2))
    #     model.add(ReLU())
    #     model.add(Conv1D(filters=hp.Int('filters4', min_value=8, max_value=64, step=8), kernel_size=hp.Int('kernel4', min_value=3, max_value=21, step=2), padding='same',activation='relu'))
        
    #     model.add(UpSampling1D(size=2))
    #     model.add(ReLU())
    #     model.add(Conv1D(filters=hp.Int('filters5', min_value=8, max_value=64, step=8), kernel_size=hp.Int('kernel5', min_value=3, max_value=21, step=2), padding='same',activation='relu'))
        
    #     model.add(UpSampling1D(size=2))
    #     model.add(ReLU())
    #     model.add(Conv1D(filters=hp.Int('filters6', min_value=8, max_value=64, step=8), kernel_size=hp.Int('kernel6', min_value=3, max_value=21, step=2), padding='same',activation='relu'))
        
    #     model.add(Dense(units=np.size(X_train, axis=1), activation="linear"))
        
    #     model.compile(optimizer=Adadelta(learning_rate = 1), loss='mse')
    #     return model
    
    # build_model(keras_tuner.HyperParameters())
    
    # # ## Hyperparameter Tuning
    # tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
    # objective="val_loss",
    # max_trials=max_trials,
    # executions_per_trial=1,
    # overwrite=True,
    # # directory=os.getcwd(),
    # directory='C:/Users/'+User+'/Documents/Python/Autoencoder',
    # project_name="Autoencoder_removal")
    # # tuner.search_space_summary()
    
    # X_train_sub, X_train_val,  = train_test_split(X_train_scaled test_size=0.3, random_state=19)
    
    # tuner.search(np.expand_dims(X_train_sub, axis=1), np.expand_dims(X_train_sub, axis=1), epochs = epochs, validation_data=(np.expand_dims(X_train_val, axis=1), np.expand_dims(X_train_val, axis=1)), verbose=2, 
    #               callbacks = [keras.callbacks.TensorBoard(log_dir="C:/Users/"+User+"/Documents/Python/Autoencoder/logs/fit")])
    # ## Type in (Anaconda) command line for tensorflow: tensorboard --logdir C:/Users/scrouse6/Documents/Python/Autoencoder/logs/fit
    # tuner.results_summary(1)
    # # best_model = tuner.get_best_models(num_models=1)
    # # best_model[0].build()
    # # best_model[0].summary()
    # # X_removed = best_model[0].predict(X_test)
    
    # best_hps = tuner.get_best_hyperparameters(5)
    # model = build_model(best_hps[0])
    
    # build_model(keras_tuner.HyperParameters())
    
    ## Used Model
    keras.utils.set_random_seed(seed)
    model = keras.Sequential()
    model.add(Conv1D(filters=112, kernel_size=21, padding='same',activation='relu', input_shape = (np.size(X_train,axis=1), 1)))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Conv1D(filters=56, kernel_size=3, padding='same',activation='relu'))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=32, kernel_size=7, padding='same',activation='relu'))
    
    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    
    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    
    model.add(Dense(units=np.size(X_train, axis=1), activation="linear"))
    
    model.compile(optimizer=Adadelta(learning_rate = 1), loss='mse')
    
    model.fit(np.expand_dims(X_train_scaled, axis=2), np.expand_dims(X_train_scaled, axis=2), epochs = epochs, verbose=0)

    t = time.time()
    X_removed_dim = model.predict(np.expand_dims(X_test_scaled, axis=2))[:,:,0]
    X_removed = scaler.inverse_transform(X_removed_dim)[:,:-(mod+1)]
    
    return X_removed, t

## Real-time autoencoder removal. One model is trained, and separate testing experiments are predicted.
def Autoencoder_removal_rt(X_train, X_test, max_trials = 32, epochs = 50, Simulated_examples = 0, seed = 19):
    if np.size(X_train, axis=1) % 8 != 0:
        mod = 7 - (np.size(X_train, axis=1) % 8)
        X_train = np.hstack((X_train, X_train[:,-2:-1]*np.ones((np.size(X_train, axis=0), (mod+1)))))
        X_test = np.hstack((X_test, X_test[:,-2:-1]*np.ones((np.size(X_test, axis=0), (mod+1)))))
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    
    ## Manual model after hyperparameter fitting
    keras.utils.set_random_seed(seed)
    model = keras.Sequential()
    model.add(Conv1D(filters=112, kernel_size=21, padding='same',activation='relu', input_shape = (np.size(X_train,axis=1), 1)))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Conv1D(filters=56, kernel_size=3, padding='same',activation='relu'))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=32, kernel_size=7, padding='same',activation='relu'))
    
    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    
    model.add(UpSampling1D(size=2))
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
    
    model.add(Dense(units=np.size(X_train, axis=1), activation="linear"))
    
    model.compile(optimizer=Adadelta(learning_rate = 1), loss='mse')
    
    model.summary()
    
    model.fit(np.expand_dims(X_train_scaled, axis=2), np.expand_dims(X_train_scaled, axis=2), epochs = epochs, verbose=0)
    
    t = time.time()
    
    X_removed_dim = np.zeros_like(X_test_scaled)
    X_removed_dim[:,:] = model.predict(np.expand_dims(X_test_scaled[:,:], axis=2))[:,:,0]
    X_removed = scaler.inverse_transform(X_removed_dim)[:,:-(mod+1)]
    
    return X_removed, t

## Calculated error assuming a gaussian distribution
def Error_estimation(X_train, X_ref, y_train, wavenumber):
    X_error_residuals = (y_train@(X_ref)-X_train)
    X_error_std = np.std(X_error_residuals)
    X_error_mean = np.mean(X_error_residuals)
    X_error_z_score = (X_error_residuals-X_error_mean)/X_error_std
    Error_limit = .50

    n = len(wavenumber)
    
    z_adjustment = np.sqrt(2)*scipy.special.erfinv(2*(Error_limit)**(1/n)-1)
    sigma_estimate = z_adjustment*X_error_std
    return sigma_estimate

def Constrained_CLS_removal(X_train, y_train, X_test, X_ref):
    wavenumber_index = np.arange((np.size(X_train,axis=1)))
    sigma_estimate = Error_estimation(X_train, X_ref, y_train, wavenumber_index)
    t = time.time()
    X_test_pyomo_sigma = X_test[:,:].copy() + sigma_estimate
    C_pyomo = np.zeros((np.size(X_test, axis=0),np.size(y_train, axis=1)))
    for k in range(np.size(X_test, axis=0)):
        try:
            model = ConcreteModel()
            model.x = Var(range(np.size(y_train,axis=1)), initialize=0.01, within=NonNegativeReals)
            ##Maximized Concentration Objective
            # model.o = Objective(expr = -sum(sum(model.x[i]*X_ref[i,j] for j in range(np.size(X_ref, axis=1))) for i in range(np.size(X_ref, axis=0))))
            ##Least Squares Objective:
            model.o = Objective(expr = sum(sum((model.x[i]*X_ref[i,j]-X_test[k,j])**2 for j in range(np.size(X_ref, axis=1))) for i in range(np.size(X_ref, axis=0))))
    
            def Fit_Rule(model,i):
                return sum(model.x[j]*X_ref[j,i] for j in range(np.size(X_ref, axis=0)))-X_test_pyomo_sigma[k,i] <=0
            model.c = Constraint(wavenumber_index, rule = Fit_Rule)
            
            solver = SolverFactory('cplex')
            if k == 0:
                results = solver.solve(model)
            else:
                results = solver.solve(model)
            C_pyomo[k,:] = np.array(list(model.x.extract_values().items()))[:,1]
            assert_optimal_termination(results)
        except:
            C_pyomo[k,:] = X_test[k,:]@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    X_removed = C_pyomo@X_ref
    return X_removed, C_pyomo, t

def BSS_removal(X_train, X_test, X_ref, n_sources, seed):
    X = np.vstack((X_train,X_test))
    n_components=np.size(X_ref, axis=0)
    ica = FastICA(n_components=n_sources, whiten = 'unit-variance',tol=1e-5,max_iter=1000, random_state=seed)
    ica.fit_transform(X.T)  # Reconstruct signals, needs the transpose of the matrix
    Aica = ica.mixing_  # Get estimated mixing matrix
    
    S0ica = (np.linalg.pinv(Aica)@X)  # Reconstruct signals, needs the transpose of the matrix
    
    # Compute MCR
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-2,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    mcrals.fit(X, ST = S0ica**2 )
    
    S0mcr = mcrals.ST_opt_;
    S0mcr_initial=S0mcr.copy()
    Amcr  = mcrals.C_opt_; 
    
    Cor = np.corrcoef(S0mcr,X_ref)
    Cor_subset = Cor[n_sources:,:n_sources]
    Column_max = np.max(Cor_subset, axis=0)
    
    sources_removed = np.empty((n_sources-n_components), dtype=int)
    for i in range(n_sources-n_components):
        sources_removed[i] = np.argmin(Column_max); Column_max[np.argmin(Column_max)] = 1
    
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    X_new=X.copy()-Xdeflation;
    X_removed = X[np.size(X_train,axis=0):,:]-Xdeflation[np.size(X_train,axis=0):,:]
    return X_removed , Cor_subset, S0mcr, S0ica #, Xdeflation, Amcr, sources_removed

def BSS_removal4(X_train, X_test, X_ref_targets, X_ref_nontargets, n_sources, seed):
    X = np.vstack((X_train, X_test))

    if X_ref_nontargets==0:
        X_ref = X_ref_targets.copy()
        n_nontargets=0
    else:
        X_ref = np.vstack((X_ref_targets, X_ref_nontargets))
        n_nontargets = np.size(X_ref_nontargets, axis=0)
        
    n_components=np.size(X_ref_targets, axis=0)
    
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref
    pca = PCA(n_components=n_sources-n_components-n_nontargets, random_state=seed)
    pca.fit(E)
    P_pca = pca.components_
    mcr_init = np.vstack((X_ref, P_pca**2))
    
    # Compute MCR
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-6,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    
    mcrals.fit(X, ST = mcr_init, st_fix=[0,1,2])
    
    S0mcr = mcrals.ST_opt_;
    S0mcr_initial=S0mcr.copy()
    Amcr  = mcrals.C_opt_; 
    
    Cor = np.corrcoef(S0mcr,X_ref)
    Cor_subset = Cor[n_sources:,:n_sources]
    Column_max = np.max(Cor_subset, axis=0)
    
    sources_removed = np.arange((n_components), (n_sources))
    
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    X_new=X.copy()-Xdeflation;
    X_removed = X[:,:]-Xdeflation[:,:]
    X_removed = X_removed[np.size(X_train,axis=0):,:]
    return X_removed #, Cor_subset, S0mcr, S0ica, Cor , P_pca, sources_removed, E#, Xdeflation, Amcr, sources_removed


def Cor_feature_selection(wavenumber,X_train,y_train,X_test,X_ref,threshold):
    ## Convert to index
    wavenumber_orig = wavenumber.copy()
    wavenumber = np.arange(len(wavenumber_orig))
    
    pearson=np.zeros((len(y_train[0,:]),len(X_train[0,:])))
    for j in range(len(y_train[0,:])):
        for i in range(len(X_train[0,:])):
            pearson[j,i]=np.corrcoef(X_train[:,i],y_train[:,j])[0,1]
    
    w=np.hstack(([wavenumber[pearson[i,:]>(threshold)] for i in range(np.size(y_train,axis=1))]))
    
    wavenumber_new=np.unique(w)
    indexfs=wavenumber_new
    indexfs=indexfs.astype(int)
    
    X_train = X_train[:,indexfs]
    X_test = X_test[:,indexfs]
    X_ref = X_ref[:,indexfs]
    wavenumber = wavenumber[indexfs]
    wavenumber_new = wavenumber_orig[wavenumber]
    return wavenumber_new, X_train, X_test, X_ref

def SNR_feature_selection(wavenumber,X_train,X_test,X_ref,sigma,SNR):
    ## Convert to index
    wavenumber_orig = wavenumber.copy()
    wavenumber = np.arange(len(wavenumber_orig))
    
    snr_range = np.max(X_train, axis=0) - np.min(X_train, axis=0)
    
    w=wavenumber[snr_range>(SNR)]
    
    indexfs = w.copy()
    indexfs=indexfs.astype(int)
    
    X_train = X_train[:,indexfs]
    X_test = X_test[:,indexfs]
    X_ref = X_ref[:,indexfs]
    wavenumber = wavenumber[indexfs]
    wavenumber_new = wavenumber_orig[wavenumber]
    return wavenumber_new, X_train, X_test, X_ref
