#Run PCA on trained DMP weights to reduce dimensionality

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os
from statistics import variance

#sys.path.append(os.path.abspath("/home/test2/Documents/Data/post_processing"))
#fileloc = '/home/test2/Documents/Data/post_processing/weights_sep_traj/modified_11_1'
def PCA_all(fileloc):
	#Import data
	df = pd.read_csv(fileloc, delimiter=' ', names=['bf1','bf2','bf3','bf4','bf5','bf6','target'])
	df.head()	
	#Scale the weights
	features = ['bf1','bf2','bf3','bf4','bf5','bf6']
	x = df.loc[:, features].values
	x_raw=x #raw_x data
	y = df.loc[:,['target']].values
	x = StandardScaler().fit_transform(x)
	df_scaled=pd.DataFrame(data = x, columns = features).head()
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents
	             , columns = ['principal component 1', 'principal component 2'])

	finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

	#Visualize
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('2 component PCA', fontsize = 20)
	targets = ['scoring', 'pivotedChop', 'normalCut','movingPivotedChop','inHandCut','dice','angledSliceR','angledSliceL']
	colors = [((0,1,0)),'g','c','m','y','b','k','r']
	for target, color in zip(targets,colors):
	    indicesToKeep = finalDf['target'] == target
	    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
	               , finalDf.loc[indicesToKeep, 'principal component 2']
	               , c = color
	               , s = 50)
	#print(targets)
	ax.legend(targets)
	ax.grid()
	#plt.show()
	ev=pca.explained_variance_ratio_
	print('explained variance in PCA_ALL', ev)
	return principalComponents, x, x_raw #PCs is 26x2 array (Z), x is scaled & standardized data matrix, x_raw is raw data matrix

#file_path='/home/test2/Documents/Data/post_processing/weights_sep_traj'
def PCA_sep_by_dim(num_dims,fileloc):		
	if num_dims==6:
		dims=['x','y','z','alpha','beta','gamma']

	elif num_dims==3:
		dims=['x','y','z']	
	all_PCs_list=[]
	x_raw_all=[]
	x_all=[]
	
	dict_PCs_all=dict.fromkeys(dims)
	dict_x_raw_all=dict.fromkeys(dims)
	dict_x_all=dict.fromkeys(dims)

	for i in range(0,len(dims)):
		print(dims[i])
		print('----------------')
		print('fileloc',fileloc)
		#Import data
		df = pd.read_csv(fileloc+'/combined_weights_'+str(dims[i])+'.txt', delimiter=' ', names=['bf1','bf2','bf3','bf4','bf5','bf6','target'])
		df.head()
		#print(df)

		#Scale the weights
		features = ['bf1','bf2','bf3','bf4','bf5','bf6']
		x = df.loc[:, features].values
		x_raw=x
		#import pdb; pdb.set_trace()
		dict_x_raw_all[dims[i]]=x_raw
		x_raw_all.append(x_raw)

		y = df.loc[:,['target']].values
		x = StandardScaler().fit_transform(x)
		x_all.append(x)
		dict_x_all[dims[i]]=x

		df_scaled=pd.DataFrame(data = x, columns = features).head()	

		pca = PCA(n_components=2)
		principalComponents = pca.fit_transform(x)

		principalDf = pd.DataFrame(data = principalComponents
		             , columns = ['principal component 1', 'principal component 2'])

		finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

		print('principal components', principalComponents)

		all_PCs_list.append(principalComponents)
		dict_PCs_all[dims[i]]=principalComponents		

		#Visualize
		#fig = plt.figure(figsize = (8,8))
		
		ev=pca.explained_variance_ratio_
		print('explained variance', ev)

		fig=plt.figure(figsize=(12,6))		 
		ax=fig.add_subplot(2,3,i+1,) 		
		ax.set_xlabel('Principal Component 1', fontsize = 8)		
	
		plt.axis([-10,15,-10,10])
		ax.set_ylabel('Principal Component 2', fontsize = 8)
		ax.set_title('2 component PCA - %s dim' %dims[i], fontsize = 10, )		

		if dims[i]=='x' or dims[i]=='y' or dims[i]=='z':
			targets = ['scoring', 'pivotedChop', 'normalCut','movingPivotedChop','inHandCut','dice','angledSliceR','angledSliceL']
			colors = [((0,1,0)),'g','c','m','y','b','k','r']

		elif dims[i]=='alpha' or dims[i]=='beta' or dims[i]=='gamma':
			targets = ['scoring', 'pivotedChop', 'normalCut','movingPivotedChop','inHandCut','dice','angledSliceR','angledSliceL']
			colors = [((0,1,0)),'g','c','m','y','b','k','r']

		for target, color in zip(targets,colors):
		    indicesToKeep = finalDf['target'] == target
		    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
		               , finalDf.loc[indicesToKeep, 'principal component 2']
		               , c = color
		               , s = 50)
		    
		    #ax.text(0.01, 0.95, 'variance explained_PC1 =%.2f' % ev[0], transform=ax.transAxes, fontsize=7,verticalalignment='top')
		    #ax.text(0.01, 0.91, 'variance explained_PC2 =%.2f' % ev[1], transform=ax.transAxes, fontsize=7,verticalalignment='top')


		ax.legend(targets, bbox_to_anchor=(-.5, .5), loc='upper center')		
		ax.grid()		

	print(len(all_PCs_list))

	return dict_PCs_all, dict_x_all, dict_x_raw_all #list of #dims x n x 2 >> length of list = 6 (# dims) 


#Reconstruct original data (26 variables in terms of 2 principal components)
#fileloc="/home/test2/Documents/Data/post_processing/combined_weights_ALL.txt"
def PCA_reconstruct_all(fileloc):
	plt.close('all')	
	PCs, x_scaled, x_raw =PCA_all(fileloc) #PCs is 26x2, this is Z , x is scaled & standardized data matrix
	cov_mat=np.cov(x_scaled.T) #Compute covariance matrix
	eig_vals, eig_vecs = np.linalg.eig(cov_mat) #Eigendecomposition on covariance matrix
	#import pdb; pdb.set_trace()
	print('Eigenvectors',eig_vecs)
	print('eigenvalues',eig_vals)

	#Choose the 1st 2 eigenvectors corresponding to the highest eigenvalues
	V=eig_vecs[0:2,:]
	V=np.transpose(V) #V is pxk matrix, i.e. 6x2
	print('V = ', V)
	V_t=np.transpose(V)
	print('V_Transposed, 2x6 matrix is ', V_t)
	print('V1 = ', V_t[0,:])
	print(V_t[0,:].shape)
	print('V2 = ', V_t[1,:])
	print(V_t[1,:].shape)

	Z=np.matmul(x_scaled,V) 
	print('Z = ', Z)
	print(Z.shape)
	Xhat=np.matmul(Z,np.transpose(V)) #XV'

	#Define mu and sigma vectors
	mu=np.mean(x_raw, axis=0)
	sigma=np.std(x_raw, axis=0)

	Xhat_raw=Xhat*sigma + mu
	print('PCs')
	print(PCs)
	print(PCs.shape)

#Output V and Z, use V as weights to run DMPs 
def PCA_reconstruct_sep_by_dim(num_dims,fileloc='/home/test2/Documents/Data/post_processing/weights_sep_traj'): 
	plt.close('all')
	if num_dims==6:
		dims=['x','y','z','alpha','beta','gamma']

	elif num_dims==3:
		dims=['x','y','z']	
	
	#Get dictionaries for PCs, raw X data, and scaled X data for each dimension (x/y/z/alpha/beta/gamma)
	dict_all_PCs, dict_x_all, dict_x_raw =PCA_sep_by_dim(num_dims,fileloc) #PCs is 26x2, this is Z , x is scaled & standardized data matrix
		
	dict_V=dict.fromkeys(dims)
	dict_Z=dict.fromkeys(dims)
	dict_mu=dict.fromkeys(dims)
	dict_sigma=dict.fromkeys(dims)

	#Calculate covariance matrix for each dimension to get V for each dimension
	for i in range(0,len(dims)):
		print(dims[i])	
		cov_mat=np.cov(dict_x_all[dims[i]].T) #Compute covariance matrix
		eig_vals, eig_vecs = np.linalg.eig(cov_mat) #Eigendecomposition on covariance matrix
		eig_vecs=eig_vecs.real
		eig_vals=np.real_if_close(eig_vals,tol=10)
		
		print('Eigenvectors',eig_vecs)
		print('eigenvalues',eig_vals)

		idx = eig_vals.argsort()[::-1]   
		eig_vals = eig_vals[idx]
		eig_vecs = eig_vecs[:,idx]
		print('________')
		print('sorted Eigenvectors',eig_vecs)
		print('________')
		print('sorted eigenvalues',eig_vals)
		print('________')

		#Choose the 1st 2 eigenvectors corresponding to the highest eigenvalues
		V=eig_vecs[:,0:2] #V is pxk matrix, i.e. 6x2	
		V_t=np.transpose(V)
		dict_V[dims[i]]=V_t #save V_t for each dimension in dictionary	
		Z=np.matmul(dict_x_all[dims[i]],V) 
		dict_Z[dims[i]]=Z #save Z for each dimension in dictionary
		Xhat=np.matmul(Z,np.transpose(V)) #XV'
		#Define mu and sigma vectors
		mu=np.mean(dict_x_raw[dims[i]], axis=0)
		dict_mu[dims[i]]=mu
		sigma=np.std(dict_x_raw[dims[i]], axis=0)
		dict_sigma[dims[i]]=sigma
		Xhat_raw=Xhat*sigma + mu		
	return dict_V, dict_Z, dict_mu, dict_sigma

def PCA_reconstruct_sep_by_dim_morePCs(num_dims,numPCs,fileloc): 
	print('numPCs',numPCs)
	plt.close('all')
	if num_dims==6:
		dims=['x','y','z','alpha','beta','gamma']
	elif num_dims==3:
		dims=['x','y','z']
	
	#Get dictionaries for PCs, raw X data, and scaled X data for each dimension (x/y/z/alpha/beta/gamma)
	dict_all_PCs, dict_x_all, dict_x_raw =PCA_sep_by_dim(num_dims,fileloc) 
	print('dict_x_raw',dict_x_raw)
	dict_V=dict.fromkeys(dims)
	dict_Z=dict.fromkeys(dims)
	dict_mu=dict.fromkeys(dims)
	dict_sigma=dict.fromkeys(dims)

	dict_var_expl=dict.fromkeys(dims)

	#Calculate covariance matrix for each dimension to get V for each dimension	
	for i in range(0,len(dims)):
		print(dims[i])

		cov_mat=np.cov(dict_x_all[dims[i]].T) #Compute covariance matrix
		eig_vals, eig_vecs = np.linalg.eig(cov_mat) #Eigendecomposition on covariance matrix
		eig_vecs=eig_vecs.real
		eig_vals=np.real_if_close(eig_vals,tol=10)
		
		print('Eigenvectors_unsorted',eig_vecs)
		print('eigenvalues_unsorted',eig_vals)

		idx = eig_vals.argsort()[::-1]   
		eig_vals_sorted = eig_vals[idx]
		eig_vecs_sorted = eig_vecs[:,idx]
		print('________')
		print('sorted Eigenvectors',eig_vecs_sorted)
		print('________')
		print('sorted eigenvalues',eig_vals_sorted)
		print('________')
		#Choose the 1st n eigenvectors corresponding to the highest eigenvalues
		V=eig_vecs_sorted[:,0:numPCs] #V is pxk matrix, i.e. 6x2

		#Calc var_expl by each PC:		
		explained_var=(eig_vals_sorted/np.sum(eig_vals_sorted))
		dict_var_expl[dims[i]]=explained_var
		V_t=np.transpose(V)
		dict_V[dims[i]]=V_t #save V_t for each dimension in dictionary
		Z=np.matmul(dict_x_all[dims[i]],V) 
		dict_Z[dims[i]]=Z #save Z for each dimension in dictionary
		Xhat=np.matmul(Z,np.transpose(V)) #XV'
		#Define mu and sigma vectors
		mu=np.mean(dict_x_raw[dims[i]], axis=0)
		dict_mu[dims[i]]=mu
		sigma=np.std(dict_x_raw[dims[i]], axis=0)
		dict_sigma[dims[i]]=sigma

		Xhat_raw=Xhat*sigma + mu		
	return dict_V, dict_Z, dict_mu, dict_sigma, dict_var_expl

def PCAcombinedweights(fileloc,numPCs): #PCA on combined weights for all dimensions (x,y,z,alpha,beta,gamma)
	all_PCs_list=[]
	x_raw_all=[]
	x_all=[]
	x_raw_all=[]
	dict_x_all=[]
	#Import data
	df = pd.read_csv(fileloc, delimiter=' ',names=['bf1','bf2','bf3','bf4','bf5','bf6','target'])
	df.head()
	features = ['bf1','bf2','bf3','bf4','bf5','bf6']
	x = df.loc[:, features].values
	x_raw=x
	x_raw_all.append(x_raw)

	#Scale the weights
	x = StandardScaler().fit_transform(x)
	x_all.append(x)
	x_raw=np.array((x_raw))
	x_all=np.squeeze(np.array((x_all)))

	#Calculate Z,V,mu,sigma
	cov_mat=np.cov(x_all.T) #Compute covariance matrix
	eig_vals, eig_vecs = np.linalg.eig(cov_mat) #Eigendecomposition on covariance matrix
	eig_vecs=eig_vecs.real
	eig_vals=np.real_if_close(eig_vals,tol=10)	

	idx = eig_vals.argsort()[::-1]   
	eig_vals = eig_vals[idx]
	eig_vecs = eig_vecs[:,idx]

	#Choose the 1st k eigenvectors corresponding to the highest eigenvalues
	V=eig_vecs[:,0:numPCs]
	V_t=np.transpose(V)
	Z=np.matmul(x_all,V) 
	
	#Define mu and sigma vectors
	mu=np.mean(x_raw, axis=0)
	sigma=np.std(x_raw, axis=0)

	return Z,V_t,mu,sigma

def PCA_sep_by_dim_and_cutType(num_dims,numPCs,fileloc,cutType): #PCA for single cut type (data separated by dimension)
	if num_dims==6:
		dims=['x','y','z','alpha','beta','gamma']

	elif num_dims==3:
		dims=['x','y','z']
	all_PCs_list=[]
	x_raw_all=[]
	x_all=[]
	
	dict_PCs_all=dict.fromkeys(dims)
	dict_x_raw_all=dict.fromkeys(dims)
	dict_x_all=dict.fromkeys(dims)

	for i in range(0,len(dims)):	
		#Import data
		df = pd.read_csv(fileloc+cutType+'_'+str(dims[i])+'.txt', delimiter=' ', names=['bf1','bf2','bf3','bf4','bf5','bf6','target'])

		#Scale the weights
		features = ['bf1','bf2','bf3','bf4','bf5','bf6']
		x = df.loc[:, features].values
		x_raw=x		
		dict_x_raw_all[dims[i]]=x_raw
		x_raw_all.append(x_raw)

		y = df.loc[:,['target']].values
		x = StandardScaler().fit_transform(x)
		x_all.append(x)
		dict_x_all[dims[i]]=x

	dict_V=dict.fromkeys(dims)
	dict_Z=dict.fromkeys(dims)
	dict_mu=dict.fromkeys(dims)
	dict_sigma=dict.fromkeys(dims)
	dict_var_expl=dict.fromkeys(dims)

	#Calculate covariance matrix for each dimension to get V for each dimension
	for i in range(0,len(dims)):
		cov_mat=np.cov(dict_x_all[dims[i]].T) #Compute covariance matrix
		eig_vals, eig_vecs = np.linalg.eig(cov_mat) #Eigendecomposition on covariance matrix
		eig_vecs=eig_vecs.real
		eig_vals=np.real_if_close(eig_vals,tol=10)	
		idx = eig_vals.argsort()[::-1]   
		eig_vals_sorted = eig_vals[idx]
		eig_vecs_sorted = eig_vecs[:,idx]
		print('________')
		print('sorted Eigenvectors',eig_vecs_sorted)
		print('________')
		print('sorted eigenvalues',eig_vals_sorted)
		print('________')
		#Choose the 1st 2 eigenvectors corresponding to the highest eigenvalues
		V=eig_vecs_sorted[:,0:numPCs] #V is pxk matrix, i.e. 6x2		
		#Calc var_expl by each PC:		
		explained_var=(eig_vals_sorted/np.sum(eig_vals_sorted))
		dict_var_expl[dims[i]]=explained_var

		V_t=np.transpose(V)
		dict_V[dims[i]]=V_t #save V_t for each dimension in dictionary	
		Z=np.matmul(dict_x_raw_all[dims[i]],V) 
		dict_Z[dims[i]]=Z #save Z for each dimension in dictionary

		#Define mu and sigma vectors
		mu=np.mean(dict_x_raw_all[dims[i]], axis=0)
		dict_mu[dims[i]]=mu
		sigma=np.std(dict_x_raw_all[dims[i]], axis=0)
		dict_sigma[dims[i]]=sigma			

	return dict_V, dict_Z, dict_mu, dict_sigma, dict_var_expl


def create_combined_weights_file_sep_by_dim(infile,outfilez,outfiley,outfilex): #Output weights from all cut types to text files separated by dimension (x/y/z)
	lines=open(infile).readlines()
	for i in range(0,len(lines)):
	    row=i+1
	    if row % 3 == 0:
	        with open(infile) as badfile, open(outfilez, 'a') as cleanfile:
	            cleanfile.writelines(lines[i])
	    elif row % 3 == 2:
	        with open(infile) as badfile, open(outfiley, 'a') as cleanfile:
	            cleanfile.writelines(lines[i])
	    elif row % 3 == 1:
	        with open(infile) as badfile, open(outfilex, 'a') as cleanfile:
	            cleanfile.writelines(lines[i])


