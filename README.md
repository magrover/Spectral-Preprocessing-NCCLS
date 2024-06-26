# Spectral-Preprocessing-NCCLS
Python scripts, functions, and data for preprocessing multicomponent vibrational spectra when unknown species may be present in the spectra

Preprocessing methods for vibrational spectroscopy (specifically Raman spectroscopy and Fourier Transform infrared spectroscopy) are included and tested as part of a study. The preprocessing methods are designed for a scenario where known species are to be quantified and references are available for these known species. However, there may be additional unknown species present in the spectrum that have unknown reference spectra. The preprocessing algorithms attempt to remove these unknown species. The methods preprocess spectra by taking high-dimensional spectra, reducing to a lower dimensional space, possibly performing feature selection or applying a constraint, and then returning the data to its original dimension.

A preprocessing algorithm titled nonnegatively constrained classical least squares (NCCLS) is introduced. Additional methods included are principal component analysis (PCA), spectra residual augmented classical least squares (SRACLS), a convolutional denoising autoencoder (CDAE), a blind source separation method utilizing principal component analysis for initial feature identification (BSS-PCA), and a blind source separation method utilizing independent component analysis for initial feature identification (BSS-ICA). 

•	Methods Folder: A folder containing the removal methods functions for application to other systems and a simple test scenario script.

o	Creation_Functions.py: Includes functions to create Gaussian peaks that resemble vibrational spectra and to create a dataset following a non-centered Latin Hypercube Sampling scheme.

o	Removal_Functions.py: Includes functions for removing unknown species from mixture spectra. Inputs for the functions include spectral data (possibly requiring concentration information), reference spectra of known species, and the data that is to be preprocessed. Spectra data is expected in the form of a 2D numpy array with different samples across the rows and different wavenumbers across the columns.

o	Removal_Verification.py: A test script demonstrating the removal functions in the Removal_Functions.py file.

•	Studies Folder: A folder containing data, functions, and scripts for testing the performance of the included removal functions in a variety of scenarios.

o	Computational Study 1.py: A script that tests the removal functions with computational data in both a real-time scenario and a batch scenario.

o	Computational Study 2.py: A script that tests the removal functions with computational data as the amount of data, overlap with unknown species, and noise levels are varied. Replicates are performed. In its current form, this code may take hours to run. 

o	Computational Study 2 Plot.py: A script that plots data recorded by the Computational Study 2.py script. 

o	3-19-24.npz: A numpy storage file for recording the data from Computational Study 2.py. This data is read by Computational Study 2 Plot.py.

o	Experimental Study.py: A script that loads, preprocesses, quantifies, and plots experimental Raman and attenuated total reflectance - Fourier transform infrared spectroscopy data.

o	Experimental Data.npz: A numpy storage file that stores the experimental data used by Experimental Study.py.

o	Creation_Functions_Study: Includes functions that are used to create computational spectroscopy data from Gaussian curves. Creates a dataset based on a non-centered Latin Hypercube Sampling scheme.

o	Removal_Functions_Study.py: Includes functions that test the preprocessing methods from Removal_Fucntions.py. These functions are used by both the computational and experimental studies.
