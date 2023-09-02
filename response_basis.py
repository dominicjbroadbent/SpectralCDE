import numpy as np
from scipy.special import eval_legendre

################
## REGRESSION ##
################

def fourier_basis(y, I):
    """Evaluate the standard Fourier basis at y up to the Ith basis function"""
    fourier_basis  = np.zeros((I, y.shape[0]))
    fourier_basis[0,  :] = 1

    max_power = int( np.floor( ( I - 1 ) / 2 ) )
    for i in range(max_power):
        fourier_basis[2*i + 1, :] = np.sqrt(2) * np.sin( 2 * np.pi * (i + 1) * y )
        fourier_basis[2*i + 2, :] = np.sqrt(2) * np.cos( 2 * np.pi * (i + 1) * y )
    if I % 2 == 0:
        fourier_basis[-1, :] = np.sqrt(2) * np.sin( I * np.pi * y )

    return fourier_basis

def cosine_basis(y, I):
    """Evaluate the standard Cosine basis at y up to the Ith basis function"""
    cosine_basis  = np.zeros((I, y.shape[0]))
    cosine_basis[0, :] = np.ones(y.shape[0])
    
    for i in range(1, I):
        cosine_basis[i, :] = np.sqrt(2) * np.cos( np.pi * i * y )
        
    return cosine_basis

def legendre_basis(y, I):
    """Evaluate the orthonormal Legendre basis at y up to the Ith basis function"""
    legendre_basis  = np.zeros((I, y.shape[0]))
    
    for i in range(I):
        orthonormal_coefficient = np.sqrt( ( 2*i + 1 ) / 2 )
        legendre_basis[i, :] = orthonormal_coefficient * eval_legendre(i, y)
    
    return legendre_basis
    
####################
## CLASSIFICATION ##
####################

def mother_wavelet(y):
    """Haar mother wavelet"""
    if ( y >= 0 ) and (y < 0.5 ):
        return 1
    elif ( y >= 1/2 ) and (y < 1 ):
        return -1
    else:
        return 0

def haar_basis(y, I):
    """Evaluate the standard Haar basis at y up to the Ith basis function"""
    # Get the vectorised mother wavelet
    haar_func = np.vectorize(mother_wavelet)
    
    # Initialise basis
    haar_basis = np.zeros((I, y.shape[0]))
    haar_basis[0,  :] = 1
    
    # Build basis
    n = 0
    k = 0
    for i in range(1, I):
        if k == 2**n - 1:
            n +=1
            k = 0
        else:
            k += 1
        haar_basis[i, :] = 2**(n/2) * haar_func( 2**n * y - k) 

    return haar_basis

def indicator_basis(y, I):
    """Evaluate the indicator basis at y up to the Ith basis function"""
    # Initialise basis
    indicator_basis = np.zeros((I, y.shape[0]), dtype = np.int64 )
    
    # Build basis
    for i in range(I):
        indicator_basis[i, :] = np.logical_and( (y > i - 1/2), (y < i + 1/2 ) ).astype(np.int64)
        
    return indicator_basis
    
    
