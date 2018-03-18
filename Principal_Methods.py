#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is main part of the first assignment in class Methods in Bioinformatics and refers to
Principal Component Analysis , with respect to its methods and algorithms.
In particular:
	  i. Implementation of conventional PCA
	 ii. Implementation of Probabilistic PCA
	iii. Implementation of Kernel PCA

"""


#==============================================================================
#   Libraries:
#==============================================================================

import numpy as np 
from random import random
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.datasets import make_circles
from sklearn.metrics import mean_squared_error



#==============================================================================
#   Principal Component Analysis:
#==============================================================================


def PCA(Arr,y,M):
	print("\n######___Principal Component Analysis___######\n")

	D = Arr.shape[0]
	N = Arr.shape[1]
	
	# --- Mean Vector:

	mean_vector = np.empty([D,1])
	for i in range(D):
		mean_vector[i] = np.mean(Arr[i,:])
	# print(mean_vector.shape)
	
	mean_d = np.repeat (mean_vector,N,axis=1)			# The mean vector with DxN shape
	X = Arr- mean_d										# Normalize the Data matrix, X: DxN

	# --- Scatter Matrix: is used to estimate the Covariance matrix of a multivariate normal distribution:
	Scatter_Matrix = np.empty([D,D])
	for i in range(X.shape[1]):
	    Scatter_Matrix += (X[:,i].reshape(D,1)).dot((X[:,i].reshape(D,1)).T)
	print('Scatter Matrix:\n', Scatter_Matrix)


	# --- Eigen Vectors/ Values:
	eig_val_sc, eig_vec_sc  = np.linalg.eig(Scatter_Matrix)

	# --- Check if Su=λu:
	for i in range(len(eig_val_sc)):
	    eigv = eig_vec_sc[:,i].reshape(1,D).T
	    np.testing.assert_array_almost_equal(Scatter_Matrix.dot(eigv), eig_val_sc[i]*eigv,decimal=6, err_msg='The eigenvector eigenvalue calculation is NOT correct.', verbose=True)

	
	# --- Rank the eigenvectors from highest to lowest corresponding eigenvalue and choose the top k eigenvectors.
	# --- Make a list of (eigenvalue, eigenvector) tuples:
	Pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

	# --- Sort the (eigenvalue, eigenvector) tuples from high to low - Using lambda function :) 
	Pairs.sort(key=lambda x: x[0], reverse=True)
	# print(len(Pairs))
	# Checking that the list is correctly sorted
	# for i in Pairs:
	#     print(i[0])
	#print(len(Pairs))

	# --- Construction of  eigenvector matrix U.

	q = input("Please enter S if you wish to use SVD to calculate array U or E for Eigendecomposition: ")
	
	if q == 'S':
		#	(i) SVD:
		U,S,V = np.linalg.svd(X, full_matrices=False)
	else:
		#	(ii) Eigendecomposition:
		U = np.empty([D,M])
		for i in range(M): 
			U[:,i] = Pairs[i][1]#.reshape(D,1)

	# print('Matrix U:\n', U)
	# print(U.shape)


	# --- Transforming the samples onto the new subspace with M- Dimension:
	Projected_Data = U.T.dot(X) 
	print("shape {}".format(Projected_Data.shape))  # Should be DxM						

	# --- Call of Plot functions:

	if len(y) == 1:
		if M == 2:
			Practical_Plots(Projected_Data)
		else:
			D_Plots(Projected_Data)
			Practical_Plots(Projected_Data)
	else:
		if M == 1:
			one_Plots(Projected_Data)
		elif M == 2:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
		else:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
			T_D_Plots(Projected_Data,y)
			
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----

#==============================================================================
#   Probabilistic Principal Component Analysis:
#==============================================================================


def PPCA(Arr,y,K):
	print("\n######___Probabilistic Principal Component Analysis___######\n")
	sigma_sq = int(input("Please enter 0 if s_square is zero: "))

	# --- Mean Vector:
	D = Arr.shape[0]
	N = Arr.shape[1]
	
	mean_vector = np.empty((D,1))
	for i in range(D):
		mean_vector[i] = np.mean(Arr[i,:])

	mean_d = np.repeat(mean_vector,N,axis=1)			# The mean vector with DxN shape
	X = Arr- mean_d										# X = x - mean(x)   X: Array of the Data  DxN 
	
	# --- Initializing the parameters:

	#sigma_sq = rand(0,1)
	W_old = np.array(np.random.rand(D,K))					# W_old: DxK
	#print("The matrix W is {}".format(W_old))
	print("The shape of matrix W is :{}".format(W_old.shape))
	W_new = np.empty((D,K))

	RMSE = mean_squared_error(W_old, W_new)**0.5   		# Root Mean Squared Error (RMSE)
	#print(RMSE)

	# --- EM Algorithm:
	Times = int(input("\nPlease assign the number of iterations:"))

	# Limit case of s^2 -> 0
	if sigma_sq == 0:
		W_all = np.empty((D,K))
		for i in range(Times):
			RMSEdiff = 1
			while RMSEdiff > 10**(-7):
				RMSEold = RMSE
				Omega = (linalg.inv((W_old.T).dot(W_old))).dot((W_old.T).dot(X))				# E step
				#print(Omega.shape)			
				W_new = (X.dot(Omega.T)).dot(linalg.inv(Omega.dot(Omega.T)))					# M step
				#print(W_new.shape)
				RMSE = mean_squared_error(W_old, W_new)**0.5
				W_old = W_new

				RMSEdiff = abs(RMSE - RMSEold)

				print(RMSEdiff)

			W_all+= W_new

	# s^2 != 0
	else:
		W_all = np.empty((D,K))
		for i in range(Times):
			sigma_sq = random() #; print(sigma_sq)
			sigma_sq_new = random()
			dif_sigma=1

			RMSEdiff = 1
			while RMSEdiff > 10**(-7) or dif_sigma > 10**(-8):
				RMSEold = RMSE
				M = (W_old.T).dot(W_old) + sigma_sq*(np.eye(K))
				E_Zn  = ((linalg.inv(M)).dot(W_old.T)).dot(X)
				E_Zn_ZnT = sigma_sq*linalg.inv(M) + E_Zn.dot(E_Zn.T)

				W_new = X.dot(E_Zn.T).dot(linalg.inv(E_Zn_ZnT)) #;print(W_new.shape)
			

				for i in range(N):
					Trace = np.trace(E_Zn_ZnT.dot((W_new).reshape(K,D)).dot(W_new))
					a_01 = (np.linalg.norm(X[:,i].reshape(1,D)))**2
					a_02 = ((E_Zn[:,i].reshape(1,K)).dot(W_new.T).dot(X[:,i].reshape(D,1))) 
					sigma_sq_new +=  np.sum(a_01 -2*a_02 +Trace)
	
				sigma_sq_new = sigma_sq_new /(N*D)
	
				RMSE = mean_squared_error(W_old, W_new)**0.5		
				dif_sigma = abs(sigma_sq - sigma_sq_new)
			
				W_old = W_new
				sigma_sq = sigma_sq_new
			
				RMSEdiff = abs(RMSE - RMSEold)
				#print(RMSE)
				print(RMSEdiff)
				print(dif_sigma)
				print("\n")

			W_all+= W_new
			
	W_mean = W_all/(Times)
	
	# --- SVD and Projection:

	U,S,V = np.linalg.svd(W_mean, full_matrices=False)

	Projected_Data = U.T.dot(X)
	# print(Projected_Data.shape)

	# --- Call of Plot functions:

	if len(y) == 1:
		if K == 2:
			Practical_Plots(Projected_Data)
		else:
			D_Plots(Projected_Data)
			Practical_Plots(Projected_Data)
	else:
		if K == 1:
			one_Plots(Projected_Data)
		elif K == 2:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
		else:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
			T_D_Plots(Projected_Data,y)

# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----

#==============================================================================
#   Kernel PCA:
#==============================================================================


def KERNEL(Arr,y,M):

	kernel = input("Construct the Kernel Matrix:\n Press G for Gaussian, P for Polynomial and T for Tangent: ")

	D = Arr.shape[0]
	N = Arr.shape[1]
	
	# --- Mean Vector:

	mean_vector = np.empty([D,1])
	for i in range(D):
		mean_vector[i] = np.mean(Arr[i,:])
	# print(mean_vector.shape)
	
	mean_d = np.repeat (mean_vector,N,axis=1)			# The mean vector with DxN shape
	X = Arr- mean_d										# Normalize the Data matrix, X: DxN

	N = X.shape[1]
	K = np.empty((N,N)) #; print(K.shape)


	# --- Constructing the Kernels:

	def Polynomial(Arr,p):
		for i in range(N):
			for j in range(N):
				K[i,j] = (1 + np.inner(Arr[:,i],Arr[:,j]))**p 

	def Tangent(Arr,delta):
		for i in range(N):
			for j in range(N):
				K[i,j] = np.tanh(np.inner(Arr[:,i],Arr[:,j]) + delta)

	def Gaussian_Kernel(Arr,gama):
		for i in range(N):
			for j in range(N):
				K[i,j] = np.exp(-gama*((((Arr[:,i] - Arr[:,j]).T).dot(Arr[:,i] - Arr[:,j]))**2))

	# --- Kernels construction:

	if kernel == 'G':
		gama = float(input("\nGive value for gama: "))		#gama = int(input("\nGive value for gama: "))
		Gaussian_Kernel(X,gama)
	elif kernel == 'P':
		p = float(input("\nGive value for p: "))			#p = int(input("\nGive value for p: "))
		Polynomial(X,p)
	else:
		delta = float(input("\nGive value for delta: "))	#delta = int(input("\nGive value for delta: "))
		Tangent(X,delta)

	One_N = np.empty((N,N)) 
	for i in range(N):
		One_N[i] = 1/N

	# --- Method:

	K_bar = K-(One_N.dot(K)) - (K.dot(One_N)) + ((One_N.dot(K)).dot(One_N))

	eig_values, eig_vectors  = np.linalg.eig(K_bar)
	Pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]

	# --- Sort the (eigenvalue, eigenvector) tuples from high to low: 
	Pairs.sort(key=lambda x: x[0], reverse=True)
	print("lenght {}".format(len(Pairs)))

	U = np.empty([N,M])
	for i in range(M): 
		U[:,i] = Pairs[i][1]#.reshape(D,1)

	# print('Matrix U:\n', U)
	# print(U.shape)

	Projected_Data = U.T.dot(K_bar) 
	# print(Projected_Data.shape)  							


	# --- Call of Plot functions:

	if len(y) == 1:
		if M == 2:
			Practical_Plots(Projected_Data)
		else:
			D_Plots(Projected_Data)
			Practical_Plots(Projected_Data)
	else:
		if M == 1:
			one_Plots(Projected_Data)
		elif M == 2:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
		else:
			one_Plots(Projected_Data)
			Theoretical_Plots(Projected_Data,y)
			T_D_Plots(Projected_Data,y)

# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----



#==============================================================================
#  Function for Subquestion (ii) of Theoretical Exercise:
#==============================================================================


def Theoretical_II(Arr01,Arr02):
	# print(Arr02.shape)
	def Eigenvalues(Arr):
		
		# --- Mean Vector:
		D = Arr.shape[0]
		N = Arr.shape[1]
	
		mean_vector = np.empty([D,1])
		for i in range(D):
			mean_vector[i] = np.mean(Arr[i,:])
		mean_d = np.repeat (mean_vector,N,axis=1)			# The mean vector with 3x80 shape	DxN	
		X = Arr- mean_d										# Normalize the join matrix X DxN

		# --- Scatter Matrix: 
		Scatter_Matrix = np.empty([D,D])
		for i in range(X.shape[1]):
			Scatter_Matrix += (X[:,i].reshape(D,1)).dot((X[:,i].reshape(D,1)).T)
		
		# --- Eigen Vectors/ Values of Scatter Matrix:
		eig_values, eig_vectors  = np.linalg.eig(Scatter_Matrix)

		val = np.ndarray.tolist(eig_values)
		sort_val = sorted(val,reverse =True)
	
		return sort_val
	

	
	sort_val = Eigenvalues(Arr01)
	sort_val2= Eigenvalues(Arr02)
	print("Eig 10:{}".format(sort_val))
	print("Eig 5:{}".format(sort_val2))
	maxx = max([max(sort_val),max(sort_val2)])
	x_val =[i for i in range(1,len(sort_val)+1)]
	
	fig, ax = plt.subplots()
	
	ax.scatter(x_val,sort_val,marker='o', color='blue', alpha=0.5, label='N=10')
	ax.scatter(x_val,sort_val2,marker='*',color='red',alpha=0.5,label='N=5')
	ax.set_ylim([0,maxx+10])
	
	plt.grid(True)
	plt.legend( numpoints=1 ,loc='upper right')
	plt.title('Eigenvalues_Plot')
	plt.savefig("Eigenvalues_Plot.png")
	plt.show()

# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----

#==============================================================================
#  Plot Functions:
#==============================================================================

# --- Theoretical Part:

def Theoretical_Plots(Arr,y):

	plt.scatter(Arr[0,y==0], Arr[1,y==0],color='red',marker='^',alpha=0.5,label='Circle_01')
	plt.scatter(Arr[0,y==1], Arr[1,y==1],color='blue',marker='o',alpha=0.5,label='Circle_02')
	plt.grid(True)
	plt.xlabel('Pca_01')
	plt.ylabel('Pca_02')
	plt.legend(numpoints =1,loc='lower right')
	plt.title('Projection')
	plt.savefig("Theoretical_01.png")
	plt.show()

def T_D_Plots(Arr,y):
	
	fig = plt.figure(figsize=(8,8))
	ax  = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10   

	ax.scatter(Arr[0,y==0], Arr[1,y==0],Arr[2,y==0],color='red',marker='^',alpha=0.5,label='Circle_01')
	ax.scatter(Arr[0,y==1], Arr[1,y==1],Arr[2,y==1],color='blue',marker='o',alpha=0.5,label='Circle_02')
	ax.grid(True)
	
	ax.set_xlabel('Pc_01')
	ax.set_ylabel('Pc_02')
	ax.set_zlabel('Pc_03')		
	
	plt.title('Projected in 3D')
	ax.legend(numpoints=1,loc='lower right')
	plt.savefig("3d Plot.png")
	plt.show()
	plt.close()

def one_Plots(Arr):
	y = [2 for i in range(500)]
	plt.plot(Arr[0,0:500],y,'o', markersize=7, color='blue', alpha=0.5, label='Circle_01')
	plt.plot(Arr[0,500:1000],y,'^', markersize=7, color='red', alpha=0.5, label='Circle_02')
	plt.grid(True)
	plt.xlabel('Pc_01')
	plt.ylabel('Pc_02')
	#plt.ylim([-10,10])
	plt.legend(numpoints=1,loc='lower right')
	plt.title('Projection in 1d')
	plt.savefig("1d Plot.png")
	plt.show()	


# --- Practical:

def Practical_Plots(Arr):
	
	plt.plot(Arr[0,0:3], Arr[1,0:3], 'o', markersize=7, color='blue', alpha=0.5, label='Baseline')
	plt.plot(Arr[0,3:27], Arr[1,3:27], '^', markersize=7, color='red', alpha=0.5, label='Normal Diet')
	plt.plot(Arr[0,27:51],Arr[1,27:51],'*', markersize=7,color='green',alpha=0.5,label='High-fat Diet')
	plt.grid(True)
	plt.xlabel('Pc_01')
	plt.ylabel('Pc_02')

	plt.legend(numpoints=1,loc='lower right')
	plt.title('Mice Projection')
	plt.savefig("Mice Projection.png")
	plt.show()


def D_Plots(Arr):
	
	fig = plt.figure(figsize=(8,8))
	ax  = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10   

	ax.plot(Arr[0,0:3],Arr[1,0:3],Arr[2,0:3], 'o', markersize=7, color='blue', alpha=0.5, label='Baseline')
	ax.plot(Arr[0,3:27],Arr[1,3:27],Arr[2,3:27], '^', markersize=7, color='red', alpha=0.5, label='Normal Diet')
	ax.plot(Arr[0,27:51],Arr[1,27:51],Arr[2,27:51],'*', markersize=7,color='green',alpha=0.5,label='High-fat Diet')
	
	ax.grid(True)
	ax.set_xlabel("Pc_01")
	ax.set_ylabel("Pc_02")
	ax.set_zlabel("Pc_03")	
	plt.title('Projected Mice 3D')
	ax.legend(numpoints=1,loc='lower right')
	plt.savefig("Mice 3d Plot.png")
	plt.show()
	plt.close()

# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
