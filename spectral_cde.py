import warnings

import numpy as np

from time import time

from tqdm.notebook import tqdm

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import kstest, uniform

from kernels import BaseKernel, Gaussian
from post_process import normalise_density, smooth, sharpen
from response_basis import fourier_basis, cosine_basis, haar_basis, indicator_basis, legendre_basis

class SpectralCDE:
    def __init__(self, X_nystrom, X, y, kernel, I_max, J_max, basis_method = 'Fourier'):
        """Initialise the spectral conditional density estimation class"""
        self.X_nystrom = X_nystrom
        self.X = X
        self.y = y
        self.kernel = kernel
        self.I_max = I_max
        self.J_max = J_max
        self.basis_method = basis_method       
            
        assert isinstance(kernel, BaseKernel), 'The kernel must be an instance of BaseKernel'
        assert basis_method in ['Fourier','Cosine', 'Haar', 'Legendre', 'Indicator'], 'The basis must be one of Fourier, Cosine, Haar, Legendre, Indicator...'
    
    ## SET-UP ##
    def kernel_matrix(self):
        """Compute the kernel matrix using X, squared distances stored to avoid reuse"""
        # self.K = self.kernel(self.X_nystrom, self.X_nystrom)
        self.K = self.kernel.re_call(self.X_nystrom)
        
    def eigendecomposition(self):
        """Compute the eigenvectors and eigenvalues of the kernel matrix and return them ordered"""
        if self.K.shape[0] >= self.J_max:
            eigenvalues, eigenvectors = eigh(self.K, check_finite = False)
        else:
            eigenvalues, eigenvectors = eigsh(self.K, k = self.J_max)
        
        # Descending order and throw away unnecessary elements
        self.eigenvalues = eigenvalues[::-1][:self.J_max]
        self.eigenvectors = eigenvectors[:, ::-1][:, :self.J_max]
        
    def randomised_eigendecomposition(self, p = 10, power_itrs = 1):
        """Algorithm 5.3 from Halko et al 2011 to compute the randomised eigendecomposition"""
        # Note that I think they do Alg 5.6 using one pass. I implement 5.3, could pursue 5.4?

        # Generate N x (n_components + p) matrix of N(0, 1) random variables
        omega = np.random.normal( 0, 1, size = self.K.shape[0] * (self.J_max + p) ).reshape(self.K.shape[0], self.J_max + p)

        # QR decomposition to find orthonormal matrix Q whose range approximates the range of A
        Y = self.K @ omega
        Q = np.linalg.qr(Y)[0]

        # Power iterations for improved accuracy (Algorithm 4.4)
        for i in range(power_itrs):
            Y_tilde = self.K.T @ Q
            Q_tilde = np.linalg.qr(Y_tilde)[0]
            Y = self.K @ Q_tilde
            Q = np.linalg.qr(Y)[0]
            
        # Form the small matrix B and compute eigendecomposition
        B = (Q.T @ self.K) @ Q
        
        if self.J_max <= B.shape[0]:
            eigenvalues, B_eigenvectors = eigh(B, check_finite = False)
        else:
            eigenvalues, B_eigenvectors = eigsh(B, k = self.J_max)
        
        # Compute eigenvectors and reverse order 
        eigenvectors = (Q @ B_eigenvectors)
        
        self.eigenvalues = eigenvalues[::-1][:self.J_max]
        self.eigenvectors = eigenvectors[:, ::-1][:, :self.J_max] 
        
    def nystrom_eigendecomposition(self):
        """
        Computes an approxfimation of the eigenvalues and eigenvectors of the full K using the Nyström method.
        See 'Using the Nyström Method to Speed Up Kernel Machines' Section 1.2
        
        X_nystrom must be a subset of X, or this will not be consistent.
        """
        assert self.X_nystrom.shape[0] >= self.J_max, 'The number of samples must be larger than or equal to the maximum number of spectral basis functions'
        
        # Compute kernel matrix of values for nystrom method
        K = self.kernel(self.X_nystrom, self.X_nystrom)
            
        # Compute eigendecomposition of sub matrix
        nystrom_eigenvalues, nystrom_eigenvectors = eigsh(K, k = self.J_max)

        # Order the Nyström eigenvalues and eigenvectors
        nystrom_eigenvalues = nystrom_eigenvalues[::-1]
        nystrom_eigenvectors = ( nystrom_eigenvectors[:, ::-1] ).T

        # Resize the eigenvalues
        self.eigenvalues = nystrom_eigenvalues * ( self.X.shape[0] / self.X_nystrom.shape[0] )

        # Compute the extrapolation matrix and get the approximate eigenvectors correctly resized
        E = self.kernel(self.X, self.X_nystrom)
        self.eigenvectors = E.dot( nystrom_eigenvectors.T ) * np.sqrt( self.X_nystrom.shape[0] / self.X.shape[0] ) * ( 1 / nystrom_eigenvalues)
        
        # Set the nystrom X to be the coefficient X for nystrom extension evaluation
        # We do this as the eigenvectors computed above are estimates of the eigenvectors of the full kernel matrix K(X, X)
        self.X_nystrom = self.X
        
    ### COMPUTING TENSOR PRODUCT COEFFS ###
    def _compute_response_basis(self, y, I):
        """Compute the response basis according to the specified method"""
        if self.basis_method == 'Fourier':
            return fourier_basis(y, I)
        elif self.basis_method == 'Cosine':
            return cosine_basis(y, I)
        elif self.basis_method == 'Haar':
            return haar_basis(y, I)
        elif self.basis_method == 'Indicator':
            return indicator_basis(y, I)
        elif self.basis_method == 'Legendre':
            return legendre_basis(y, I)
    
    def _compute_spectral_basis(self, x, J):
        """Evaluate the spectral basis defined by the Nystrom extension at x up to the Jth basis function"""
        assert x.ndim == 2, 'The evaluation array x must have two dimensions'
        
        spectral_basis = self.kernel(x, self.X_nystrom).dot( self.eigenvectors[:, :J] )
        spectral_basis *= ( np.sqrt( self.X_nystrom.shape[0] ) / self.eigenvalues[:J] )
        return spectral_basis.T 
    
    def compute_coefficients(self):
        """Compute Monte Carlo estimates of the spectral CDE coefficients"""
        # Evaluating Fourier basis functions
        response_basis = self._compute_response_basis(self.y, self.I_max)
        
        # Evaluating spectral basis functions
        if np.array_equal(self.X_nystrom, self.X):
            # spectral_basis = ( self.eigenvectors.T @ self.K ) * ( np.sqrt(self.X.shape[0]) / self.eigenvalues[:self.J_max] )[:, np.newaxis]
            spectral_basis = ( self.eigenvectors.T * np.sqrt(self.X.shape[0]) ) # Equivalent to above using Kv = lv
        else:
            spectral_basis = self._compute_spectral_basis(self.X, self.J_max)

        # Computing Monte Carlo coefficient estimates...
        self.coefficients = ( response_basis[:, np.newaxis, :] * spectral_basis[np.newaxis, :, :] ).mean(axis = 2)
    
    ## EVALUATING AT NEW POINTS ##
    def _evaluate_array(self, x, y, I, J):
        """
        Evaluate the spectral conditional density estimate at pairs of y's given x's
        using I response basis functions and J spectral basis functions, return the I x J x N array of values
        """        
        # Compute the fourier and spectral basis evaluated at each of the y's and at x respectively
        response_basis = self._compute_response_basis(y, I)
        spectral_basis = self._compute_spectral_basis(x, J)
        
        # Evaluate the density estimate at each (x, y) pair and for each I, J without summing, returns I x J x N array
        evals = ( response_basis[:, np.newaxis, :] * spectral_basis[np.newaxis, :, :] )
        evals *= self.coefficients[:I, :J, np.newaxis]
        return evals
    
    def evaluate(self, x, y, I, J):
        """
        Evaluate the spectral conditional density estimate at pairs of y's given x's
        using I response basis functions and J spectral basis functions, returns the vector from the I,J plane-summed array of evaluations
        """
        return self._evaluate_array(x, y, I, J ).sum(axis = 0).sum(axis = 0)
                    
    ## MODEL TRAINING ##    
    def _compute_loss_array(self, X_val, y_val):
        """Return a I_max by J_max array of the approximate integrated squared loss for each choice of I and J"""
        # Evaluate the spectral basis at the validation points
        spectral_basis = self._compute_spectral_basis(X_val, self.J_max) # J x n_val

        # # J x J array of Monte Carlo approximations to the inner products (denoted as W hat in the paper)
        inner_prods = ( spectral_basis[np.newaxis, :, :]  * spectral_basis[:, np.newaxis, :] ).mean(axis = 2) 
        
        #  I x J x J array of beta-beta-W multiplications
        sum_prods = ( ( self.coefficients[:, :, np.newaxis] * self.coefficients[:, np.newaxis, :] ) * inner_prods )
        
        #  I x J Array of the triple summed values (wasted work here but not sure how to do this faster without for loop)
        #integral_term = np.cumsum(np.cumsum(np.cumsum(sum_prods, axis = 0), axis = 1), axis = 2).diagonal(axis1 = 1, axis2 = 2)
        integral_term = sum_prods.cumsum(axis = 1).cumsum(axis = 2).diagonal(axis1 = 1, axis2 = 2).cumsum(axis = 0)

        # Array of unsummed conditional density evaluations for every combination of I and J up to their max
        evaluation_array = self._evaluate_array(X_val, y_val, self.I_max, self.J_max) # I x J x n_val
        
        # Second term of loss function I x J
        expectation_term = np.cumsum(np.cumsum(evaluation_array.mean(axis = 2), axis = 0), axis = 1)

        loss = integral_term - 2 * expectation_term
        return loss
    
    ## CROSS VALIDATION ##
    def cross_validate(self, X_val, y_val, params, verbose, random = [False, 10, 1], nystrom = False):
        """Tune the conditional density estimate parameters"""
        if random[0] and nystrom:
            raise ValueError('Randomised eigendecomposition and Nystrom method cannot both be used')
        loss = np.zeros( ( params.shape[0], self.I_max, self.J_max ) )
        
        kernel_time = 0
        eigendecomp_time = 0
        coefficient_time = 0
        loss_time = 0
        
        for i in tqdm(range(params.shape[0]), disable = not verbose):
            # Compute kernel matrix
            t0 = time()
            self.kernel.set_params(params[i])
            self.kernel_matrix()
            kernel_time += time() - t0
            
            # Compute eigendecomposition
            t0 = time()
            if random[0]:
                self.randomised_eigendecomposition(random[1], random[2])
            elif nystrom:
                self.nystrom_eigendecomposition()
            else:
                self.eigendecomposition()
            eigendecomp_time += time() - t0
            
            # Compute coefficients
            t0 = time()
            self.compute_coefficients()
            coefficient_time += time() - t0
            
            # Compute validation loss at each I and J combination
            t0 = time()
            loss[i, :, :] = self._compute_loss_array(X_val, y_val)
            loss_time += time() - t0
            
        if verbose:
            print(f'Kernel computation time: {round(kernel_time, 3)}, Eigendecomposition time: {round(eigendecomp_time, 3)},\n\
Coefficient computation time: {round(coefficient_time, 3)}, Loss computation time: {round(loss_time, 3)} ')
            
        return loss
    
    ## POST PROCESS THE CONDITIONAL DENSITY ESTIMATES ##
    def post_process(self, estimates, bin_size, delta = 0, alpha = 1):
        """Post process the conditional density estimate"""
        # Normalise the density
        estimates = normalise_density(estimates, bin_size)
        
        if delta != 0:
            # Smooth the density and then renormalise
            estimates = smooth(estimates, bin_size,  delta)
            estimates = normalise_density(estimates, bin_size)
        if alpha != 1:
            # Sharpen and renormalise the density
            estimates = sharpen(estimates, bin_size, alpha)
            estimates = normalise_density(estimates, bin_size)
            
        return estimates
    
    ## GOODNESS OF FIT TESTS ##
    def goodness_of_fit_test(self, X_test, y_test,
                             I, J,
                             y_eval_min = 0, y_eval_max = 1,
                             grid_size = 2500,
                             delta = 0, alpha = 1,
                             verbose = False):
    
        y_linspace = np.linspace(y_eval_min, y_eval_max, grid_size)
        bin_size = y_linspace[1] - y_linspace[0]
        
        # For each test point and response pair, approximate the area under the estimated conditional density 
        # "to the left" of the observed test response value
        U = np.zeros(X_test.shape[0])
        for i in tqdm(range(U.shape[0]), disable = not verbose):
            cde = self.evaluate(X_test[[i], :], y_linspace, I, J)
            cde = self.post_process(cde, bin_size, delta)
            U[i] = bin_size * cde[ y_linspace < y_test[i] ].sum()
            
        # Perform KS test
        ks_stat, p_value = kstest(U, 'uniform')
        
        # Print results
        if verbose:
            print(f'KS test statistic: {round(ks_stat, 4)}')
            print(f'p-value: {round(p_value, 10)}')
            if p_value < 0.05:
                print('Reject the null hypothesis (at the 95% level) that the data is from a U(0, 1) distribution.')
            else:
                print('Fail to reject the null hypothesis that the data is from a U(0, 1) distribution.')
        
        return p_value, ks_stat
