# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 05:04:24 2015

@author: maria
"""

import numpy as np
import statsmodels
import scipy as sc

class MultivariateAutoregressiveModel(object):
    def __init__(self,timeseries_matrix):
        '''
        The input matrix has to have dimensions Txd,
        where T is the number of time points and d is 
        the number of variables.
        '''
        self.timeseries_matrix = timeseries_matrix 
        
    def compose_feature_matrix(self,p):
        N = self.timeseries_matrix.shape[0]
        d = self.timeseries_matrix.shape[1]
        X = self.timeseries_matrix[0:(N-p+1),:]
        for lag in range(1,p):
            X=hstack((X, self.timeseries_matrix[lag:(N-p+lag+1),:]))  
        reordering={}    
        for lag in range(0,p):
            start=lag*d
            stop=start+d
            reordering[lag]=X[:,start:stop]

        X=reordering[p-1]  
        for lag in range(p-2,-1,-1):
            X=hstack((X,reordering[lag]))    
    
        rows=X.shape[0]
        X=X[0:rows-1,:]        
        
        return X
        
    def maximum_likelihood_fit(self,p):
        '''
        Should I attach the weights and y_pred to the object?
        '''
        X = self.compose_feature_matrix(p)
        Y = self.timeseries_matrix[p:,:]        
        inverse_X = np.linalg.pinv(X)
        ml_weights = np.dot(inverse_X,Y)
        ml_y_pred = np.dot(X,ml_weights)
        
        residuals = Y-ml_y_pred
        residual_sum_of_sq=np.dot(residuals.T,residuals)
        noise_cov = np.cov(residuals)        
        
        return ml_weights, ml_y_pred, residual_sum_of_sq, noise_cov
        
    def ridge_regression_fit(self,p,alpha):
        X = self.compose_feature_matrix(p)
        Y = self.timeseries_matrix[p:,:]
        G = alpha*np.eye(X.shape[0],X.shape[1])
        inverse_XG = np.linalg.pinv(X+G)
        ridge_weights = np.dot(inverse_XG,Y)
        ridge_y_pred = np.dot(X,ridge_weights)
        
        residuals = Y-ridge_y_pred
        residual_sum_of_sq=np.dot(residuals.T,residuals)
        noise_cov = np.cov(residuals)        
        
        return ridge_weights, ridge_y_pred, residual_sum_of_sq, noise_cov
        
    def information_criteria(self, residual_sum_of_sq,p):
        '''This function computes information criteria
        that can be used for model order selection.'''
        T = self.timeseries_matrix.shape[0]
        n = self.timeseries_matrix.shape[1]
        sigma_p = (1./T)*np.sum(np.diag(residual_sum_of_sq))               
        AIC = np.log(np.abs(sigma_p)) + (2./T)*p*n**2
        BIC = np.log(np.abs(sigma_p)) + (float(np.log(T))/T)*p*n**2
        HQIC = np.log(np.abs(sigma_p)) + (float(np.log(np.log(T)))/T)*p*n**2
        
        return {
            'aic' : AIC,
            'bic' : BIC,
            'hqic' : HQIC,
        }
        
class Connectivity(object):
    '''
    This class is copied from SCoT, except
    weight vector is transposed.
    '''
    def __init__(self, b, c=None, nfft=512):
        b = np.asarray(b)
        #Transpose coefficients according to this implementation
        b=b.T
        (m, mp) = b.shape
        print b.shape
        p = mp // m
        if m * p != mp:
            raise AttributeError('Second dimension of b must be an integer multiple of the first dimension.')

        if c is None:
            self.c = None
     

        self.b = np.reshape(b, (m, m, p), 'c')
        self.m = m
        self.p = p
        self.nfft = nfft
        
    def A(self):
        """ Spectral VAR coefficients
        .. math:: \mathbf{A}(f) = \mathbf{I} - \sum_{k=1}^{p} \mathbf{a}^{(k)} \mathrm{e}^{-2\pi f}
        """
        return fft(np.dstack([np.eye(self.m), -self.b]), self.nfft * 2 - 1)[:, :, :self.nfft]

    def PDC(self):
        """ Partial directed coherence
        .. math:: \mathrm{PDC}_{ij}(f) = \\frac{A_{ij}(f)}{\sqrt{A_{:j}'(f) A_{:j}(f)}}
        References
        ----------
        L. A. Baccalá, K. Sameshima. Partial directed coherence: a new concept in neural structure
        determination. Biol. Cybernetics 84(6):463-474, 2001.
        """
        A = self.A()
        return np.abs(A / np.sqrt(np.sum(A.conj() * A, axis=0, keepdims=True)))
        
class Simulation(object):
    def __init__(self, model_order,parameters,nr_of_variables, time_series_length):
        self.model_order =  model_order
        self.parameters  = parameters
        self.nr_of_variables  = nr_of_variables
        self.time_series_length = time_series_length
        
    def univariate_simulation(self):
        noise_var = 1**2
        noise = sqrt(noise_var)*np.random.rand(1,self.time_series_length)
        b = np.ones(1+len(self.parameters))
        a = np.squeeze(np.array([[1]+self.parameters]))
        y = sc.signal.lfilter(b,a,noise,axis=0)
        return y
        

##########################################

#sim = Simulation(5,[-1.8517,1.3741,0.1421,-0.6852,0.3506],1,640)
#y = sim.univariate_simulation()

########################################## 
def generate_next_time_point(lag_tensor,time_window):
    components=[]
    for j in range(0,lag_tensor.shape[0]):
        components.append(np.dot(lag_tensor[j,:,:], time_window[j,:]))
    prediction = np.sum(components,axis=0)

    return prediction
    
def initial_simulation():
    initial_condition = np.random.rand(2,2)
    initial_condition = np.array([[1,1],[1,1]])
    act_act = np.array([2,2])
    inh_act = np.array([-0.5,2])
    inh_inh = np.array([-0.5,-0.5])
    act_inh = np.array([2,-0.5])
    
    lag_tensor=np.array([[[act_act,inh_act]],[[inh_inh,act_inh]]])
    for j in range(0,100):
        next_time_point= generate_next_time_point(lag_tensor,initial_condition[-2:,:])
        initial_condition=np.append(initial_condition,next_time_point,axis=0)
    print initial_condition
initial_simulation()
#from scipy import io
#
#'''
#fMRI data obtained from
#http://www.nimh.nih.gov/labs-at-nimh/research-areas/clinics-and-labs/chp/research-articles.shtml
#It's data from healthy controls and patients with childhood onset schizophrenia.
#'''
#
#data=io.loadmat('NV_COS_data.mat')
#timeseries_mat=data['NVdata']['timeseries'][0][0][0][0]
#print timeseries_mat.shape
#timeseries_mat = timeseries_mat.T
#print timeseries_mat.shape
#    
#for j in range(0,timeseries_mat.shape[0]):
#    timeseries_mat[j,:]=np.subtract(timeseries_mat[j,:],np.mean(timeseries_mat[j,:]))
# 
#autoregr= MultivariateAutoregressiveModel(timeseries_mat)
#weights, y_pred, ssr, noise_cov = autoregr.ridge_regression_fit(2,alpha=0) 
#
#conn = Connectivity(weights,noise_cov)
#PDC = conn.PDC()
#
#'''
#Experimenting with stasmodels
#'''
#
##model = statsmodels.tsa.vector_ar.var_model.VAR(timeseries_mat[:,0:128])
##results=model.fit(2)
##print results.bic
#
#import statsmodels
#import statsmodels.api as sm
#from statsmodels.sandbox.tools import pca
#
#
##Take PCA, because there are more brain regions
##than lenght of time series
#xred, fact, eva, eve  = pca(timeseries_mat, keepdim=3, normalize=1)
##fact_wconst = sm.add_constant(fact[:,:10], prepend=False)
#print eva
#
##fact_wconst = sm.add_constant(fact, prepend=False)
#z=np.real(fact)
#model = statsmodels.tsa.vector_ar.var_model.VAR(z)
#results=model.fit(2)
#print results.bic
#
#
#autoregr= MultivariateAutoregressiveModel(fact)
#weights, y_pred, residual, noise_cov = autoregr.ridge_regression_fit(3,alpha=4) 
#print autoregr.information_criteria(noise_cov,3)
