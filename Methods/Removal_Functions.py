# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:01:55 2023

@author: scrouse6
"""

import numpy as np
from scipy.special import erfinv
from sklearn.decomposition import FastICA, PCA 
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from pyomo.environ import *


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.optimizers import Adadelta


## Calculated error assuming a gaussian distribution
def Error_estimation(X_train, X_ref, y_train):
    n = np.size(X_train, axis=1)
    
    ## Calculate residuals of CLS and calculate mean and standard deviation
    X_error_residuals = (y_train@(X_ref)-X_train)
    X_error_std = np.std(X_error_residuals)
    X_error_mean = np.mean(X_error_residuals)
    X_error_z_score = (X_error_residuals-X_error_mean)/X_error_std
    
    ## Define the percentage of spectra that will have at least one residual greater than "sigma estimate" (0.5 represents half of all spectra and is normal)
    Error_limit = .50
    
    ## Calculate expected error bounds (in z-score)
    z_adjustment = np.sqrt(2)*erfinv(2*(Error_limit)**(1/n)-1)
    
    ## Convert z-score to domain of the error
    sigma_estimate = z_adjustment*X_error_std
    
    return sigma_estimate

def NCCLS(X_train, y_train, X_test, X_ref):
    ## Use Error_estimation function to find an estimate of error from available data
    sigma_estimate = Error_estimation(X_train, X_ref, y_train)
    
    ## Find index values for wavenumbers
    wavenumber_index = np.arange((np.size(X_train,axis=1)))
    
    ## Create modified test data that has been shifted by the sigma estimate
    X_test_pyomo_sigma = X_test.copy() + sigma_estimate
    
    ## Create numpy array for use in Pyomo optimization
    C_pyomo = np.zeros((np.size(X_test, axis=0),np.size(y_train, axis=1)))
    
    ## Iterate through each spectrum in X_test and perform optimization on each spectrum individually
    for k in range(np.size(X_test, axis=0)):
        ## First, try the NCCLS model
        try:
            ## Create the Pyomo model
            model = ConcreteModel()
            
            ## Introduce the variable x, which represents concentration here (and has the dimension of concentration)
            model.x = Var(range(np.size(y_train,axis=1)), initialize=0.01, within=NonNegativeReals)
           
            ## Objective function is a least squares fit between fit spectra (CK) and mixture spectra (A)
            model.o = Objective(expr = sum(sum((model.x[i]*X_ref[i,j]-X_test[k,j])**2 for j in range(np.size(X_ref, axis=1))) for i in range(np.size(X_ref, axis=0))))
    
            ## Create constraints so that the (adjusted by sigma estimate) process spectra is greater than the fit spectra (CK) at every wavenumber; there will be as many constraints as wavenumbers in a spectrum
            def Fit_Rule(model,i):
                return sum(model.x[j]*X_ref[j,i] for j in range(np.size(X_ref, axis=0)))-X_test_pyomo_sigma[k,i] <=0
            model.c = Constraint(wavenumber_index, rule = Fit_Rule)
            
            ## Indicate the solver used, in this case IBM's cplex solver for quadratic programming problems
            solver = SolverFactory('cplex')
            
            ## Solve the model
            results = solver.solve(model)
    
            ## Place the results in the numpy array C_pyomo
            C_pyomo[k,:] = np.array(list(model.x.extract_values().items()))[:,1]
            
            ## Print performance of the model
            assert_optimal_termination(results)
            
        ## If NCCLS fails to provide a solution (usually because there are no nonnegative C values that satisfy the constraints, likely due to areas of the spectrum with low SNR or baselining issues)
        except:
            ## Perform uncsontrained CLS preprocessing as a fail-safe
            C_pyomo[k,:] = X_test[k,:]@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
            print('NCCLS did not find a solution')
    
    ## Remove non-target species by using calculated C matrix with original references        
    X_removed = C_pyomo@X_ref
    
    return X_removed

def PCA_removal(X_train, X_test, n_species):
    ## Fit training data with PCA
    pca_transform = PCA(n_components=n_species)
    X_train_transformed = pca_transform.fit_transform(X_train); X_train_pca = pca_transform.inverse_transform(X_train_transformed)
    
    ## Transform process data with only a few (n_species) components
    X_test_transformed = pca_transform.transform(X_test); X_test_pca = pca_transform.inverse_transform(X_test_transformed)
    
    ## Invert the transformed process data back to original space
    X_removed = X_test_pca
    
    return X_removed

def SRACLS_removal(X_train, X_ref, X_test, n_species):
    ## Creation of CLS residuals
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref

    ## Performing PCA on residuals
    pca = PCA(n_components=n_species)
    pca.fit(E)
    P_pca = pca.components_
    
    ## Extending reference spectra to include PCA source(s)
    X_ref_pca = np.vstack((X_ref, P_pca))

    ## Recreating spectra with additional non-target component removed through CLS
    Y_pca_resid = X_test@X_ref_pca.T@np.linalg.inv(X_ref_pca@X_ref_pca.T)
    X_removed = Y_pca_resid[:,:np.size(X_ref, axis=0)]@X_ref_pca[:np.size(X_ref, axis=0),:]
    
    return X_removed

def Autoencoder_removal(X_train, X_test, epochs = 400):
    # Extends domain to be divisible by 2^3 (8) for convolutional autoencoder
    if np.size(X_train, axis=1) % 8 != 0:
        mod = 7 - (np.size(X_train, axis=1) % 8)
        X_train = np.hstack((X_train, X_train[:,-2:-1]*np.ones((np.size(X_train, axis=0), (mod+1)))))
        X_test = np.hstack((X_test, X_test[:,-2:-1]*np.ones((np.size(X_test, axis=0), (mod+1)))))
    
    ## Apply standard normal variate scaling before implementing autoencoder
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ## Create keras model with convolutional layers
    model = Sequential()
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
    
    ## Final layer as a dense layer with linear activation function showed the best performance
    model.add(Dense(units=np.size(X_train, axis=1), activation="linear"))
    
    model.compile(optimizer=Adadelta(learning_rate = 1), loss='mse')
    model.summary()
    
    ## Fit the model
    model.fit(np.expand_dims(X_train_scaled, axis=2),np.expand_dims(X_train_scaled, axis=2), epochs = epochs, verbose=0)
    
    ## Apply the model to process data
    X_removed_dim = model.predict(np.expand_dims(X_test_scaled, axis=2))[:,:,0]
    
    ## Invert scaling and remove additional wavenumbers created to facilitate convolution
    X_removed = scaler.inverse_transform(X_removed_dim)[:,:-(mod+1)]
    
    return X_removed

def BSS_ICA(X_train, X_test, X_ref, n_sources):
    n_components=np.size(X_ref, axis=0)
    
    ## Use all mixture spectra to identify sources
    X = np.vstack((X_train,X_test))
    
    ## Perform ICA on all mixture spectra
    ica = FastICA(n_components=n_sources, whiten = 'unit-variance',tol=1e-5,max_iter=1000)
    ica.fit_transform(X.T) 
    
    ## Extract estimates of pure-component sources and concentrations from ICA
    Aica = ica.mixing_  
    S0ica = (np.linalg.pinv(Aica)@X)
    
    ## Perform MCR-ALS with a nonnegativity constraint
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-2,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    mcrals.fit(X, ST = S0ica**2 )
    
    ## Extract estimates of pure-component sources and concentrations from MCR
    S0mcr = mcrals.ST_opt_;
    Amcr  = mcrals.C_opt_; 
    
    ## Match target species with available references based on correlation
    Cor = np.corrcoef(S0mcr,X_ref)
    Cor_subset = Cor[n_sources:,:n_sources]
    Column_max = np.max(Cor_subset, axis=0)
    sources_removed = np.empty((n_sources-n_components), dtype=int)
    for i in range(n_sources-n_components):
        sources_removed[i] = np.argmin(Column_max); Column_max[np.argmin(Column_max)] = 1
    
    ## Calculate contributions of non-target sources
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    
    ## Subtract non-target contributions from all spectra (and select process spectra only)
    X_removed = X[np.size(X_train,axis=0):,:]-Xdeflation[np.size(X_train,axis=0):,:]
    
    return X_removed

def BSS_PCA(X_train, X_test, X_ref, n_sources):
    n_components=np.size(X_ref, axis=0)

    ## Use all mixture spectra to identify sources
    X = np.vstack((X_train, X_test))

    ## Identify CLS residuals on spectra
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref
    
    ## Perform PCA on residuals
    pca = PCA(n_components=n_sources-n_components)
    pca.fit(E)
    P_pca = pca.components_
    
    ## Create starting reference spectra for MCR-ALS; includes original references and squared (nonnegative) PCA-identified sources
    mcr_init = np.vstack((X_ref, P_pca**2))
    
    ## Perform MCR-ALS, adjust st_fix to indicate how many reference spectra there are (3 in this example)
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-6,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    mcrals.fit(X, ST = mcr_init, st_fix=[0,1,2])
    
    ## Extract estimates of pure-component sources and concentrations from MCR
    S0mcr = mcrals.ST_opt_;
    Amcr  = mcrals.C_opt_; 
    
    ## Match target species with available references based on output from MCR-ALS
    sources_removed = np.arange((n_components), (n_sources))
    
    ## Calculate contributions of non-target sources
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    
    ## Subtract contrbutions of non-target species
    X_removed = X - Xdeflation
    
    ##Isolate the process data
    X_removed = X_removed[np.size(X_train,axis=0):,:]
    
    return X_removed

def Cor_feature_selection(wavenumber,X_train,y_train,X_test,X_ref,cor_threshold):
    ## Convert wavenumbers to index values
    wavenumber_orig = wavenumber.copy()
    wavenumber = np.arange(len(wavenumber_orig))
    
    ## Find Pearson Correlation between inputs (X) and outputs (y)
    pearson=np.zeros((len(y_train[0,:]),len(X_train[0,:])))
    for j in range(len(y_train[0,:])):
        for i in range(len(X_train[0,:])):
            pearson[j,i]=np.corrcoef(X_train[:,i],y_train[:,j])[0,1]
    
    ## Select wavenumbers that meet a certain threshold for correlation
    w=np.hstack(([wavenumber[pearson[i,:]>(threshold)] for i in range(np.size(y_train,axis=1))]))
    
    ## Manage duplicates
    wavenumber_new=np.unique(w)
    
    indexfs=wavenumber_new
    indexfs=indexfs.astype(int)
    
    ## Perform feature selection
    X_train = X_train[:,indexfs]
    X_test = X_test[:,indexfs]
    X_ref = X_ref[:,indexfs]
    wavenumber = wavenumber[indexfs]
    wavenumber_new = wavenumber_orig[wavenumber]
    
    return wavenumber_new, X_train, X_test, X_ref

def SNR_feature_selection(wavenumber,X_train,X_test,X_ref,SNR):
    ## Convert to index
    wavenumber_orig = wavenumber.copy()
    wavenumber = np.arange(len(wavenumber_orig))
    
    ## Find signal range at every wavenumber
    snr_range = np.max(X_train, axis=0) - np.min(X_train, axis=0)
    
    ## Select wavnumbers that meet a signal range above a certain threshold
    w=wavenumber[snr_range>(SNR)]
    
    indexfs = w.copy()
    indexfs=indexfs.astype(int)
    
    ## Perform feature selection
    X_train = X_train[:,indexfs]
    X_test = X_test[:,indexfs]
    X_ref = X_ref[:,indexfs]
    wavenumber = wavenumber[indexfs]
    wavenumber_new = wavenumber_orig[wavenumber]
    
    return wavenumber_new, X_train, X_test, X_ref
