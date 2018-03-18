#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the first assignment in class Methods in Bioinformatics and refers to
Principal Component Analysis , with respect to its methods and algorithms.

This particular script is the main call for the methods of Principal_Methods.py script 
which contains the implementation of the algorithms.

"""


#==============================================================================
#   Libraries:
#==============================================================================

import time
import os
from Principal_Methods import*

#==============================================================================
#   Calls:
#==============================================================================


Problem = input("Choose either Theoretical or Practical Problem.\n Enter A or B for Theoretical or Practical respectively:")

if Problem == 'A':
	print("****____Theoretical Problem has been selected____****\n")

	Sub_quest = input("Choose Subquestion for the Theoretical part\n Enter A for Subquestion (i) and B for Subquestion (ii): ")

	if Sub_quest == 'A':
		print("****____Theoretical Part: Subquestion (i)____****")
		
		Method = input("Choose Method: Enter PCA for Principal Component Analysis,\nEM for Probabilistic PCA and \nKERNEL for Kernel Method: ")
		Dimension = int(input("Give Dimensionality of Projection\n Notice that for either PCA or EM should be 1 or 2:  "))
		M = Dimension
		
		# ---  Data given for the theoretical problem . Subquestion (i) with different noises:
		
		n = input("Enter A for 0.05 noise and B for noise 3: ")
		if n == 'A':
			X, y = make_circles(n_samples=1000,noise=0.05, factor=0.3)
		else:
			X, y = make_circles(n_samples=1000,noise=3, factor=0.3)
		
		Data = X.T
		
		if Method == 'PCA':
			PCA(Data,y,M)
		elif Method == 'EM':
			PPCA(Data,y,M)
		else:
			KERNEL(Data,y,M)
	else:
		print("****____Theoretical Part: Subquestion (ii)____****\n")
		
		# ---  Data given for the theoretical problem . Subquestion (ii):
		
		mean = np.array([1,1,1,1,1,1,1,1,1,1])#; print(mean.shape)
		cov  = np.eye((10))#; print(cov.shape)
		
		Xa = np.random.multivariate_normal(mean,cov,10)
		Xb = Xa[:,0:5]

		Theoretical_II(Xa,Xb)

else:
	y = [0]
	print("****____Practical Problem has been selected____****\n")

	print("\nData set should be downloaded automatically and the process shall begin.\n")


	if not os.path.exists("Final.txt"):

		Filename = os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')

		os.system('gunzip <GDS6248.soft.gz> Data_set.txt')
		os.system('grep -i ILMN Data_set.txt > Data.txt')
		os.system('cut -f3- Data.txt > Final.txt')
	else:
		print('Skipping file download, Data file exists...')

	X = np.loadtxt("Final.txt")			## Constructing the Data Array
	print(X.shape)

	Method = input("Choose Method: Enter PCA for Principal Component Analysis, EM for Probabilistic PCA and KERNEL for Kernel Method: ")
	Dimension = int(input("Give Dimensionality of Projection:"))
	M = Dimension

	if Method == 'PCA':
		PCA(X,y,M)
	elif Method == 'EM':
		PPCA(X,y,M)
	else:
		KERNEL(X,y,M)

	# --- PCA with Built-in python class 
	
	from sklearn.decomposition import PCA as pca
	
	n_components = Dimension
	my_pca = pca(n_components)
	
	Projected_Data = my_pca.fit_transform(X.T).T
	
	if Dimension == 2:
		Practical_Plots(Projected_Data)
	else:
		D_Plots(Projected_Data)

print("\n\n*****____End of Process____*****\n\n")


Time = time.process_time()
print(Time)