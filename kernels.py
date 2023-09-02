import numpy as np
from scipy.spatial.distance import cdist

class BaseKernel:
    def __init__(self):
        """Initialise the kernel instance"""
        pass
    
    def __call__(self, x, y):
        """Evaluate the kernel function at (x, y)"""
        raise NotImplementedError()

class Gaussian(BaseKernel):
    def __init__(self, lengthscale = 0.5):
        """Initialise the Gaussian kernel"""
        self.lengthscale = lengthscale
        super().__init__()
        
    def __call__(self, x, y):
        """Evaluate the kernel function at (x, y)"""
        sq_dists = cdist(x, y, metric = 'sqeuclidean')
        return np.exp( - 0.5 * sq_dists / self.lengthscale)
    
    def set_params(self, new_lengthscale):
        """Set new kernel hyperparameters"""
        self.lengthscale = new_lengthscale

    def median_heuristic(self, X, m = None, verbose = False):
        """Compute the median heuristic with a subsample of m points, then set the kernel lengthscale"""        
        # Subsample the array
        if m is not None:
            assert m <= X.shape[0], 'm must be lesser than or equal to the data size'
            X = X[np.random.choice(X.shape[0], m, replace = False), :]
        
        # Compute squared distances and free up memory
        sq_dists = cdist(X, X, metric = 'sqeuclidean')
        del X 
        
        # Don't take into account zero distances
        np.fill_diagonal(sq_dists, np.nan)
        
        # Unravel the array and remove the nans
        sq_dists = np.ravel( sq_dists )
        sq_dists = sq_dists[~np.isnan(sq_dists)]
        
        # Compute the median heuristic (two definitions)
        #median_herusitic = np.sqrt( np.median( sq_dists ) / 2 )
        median_herusitic = np.sqrt( np.median( sq_dists ))
        
        # Set the lengthscale to be the median heuristic
        if verbose:
            print(f'Median heuristic = {median_herusitic}, setting lengthscale parameter...')
        self.set_params(median_herusitic)
        
    def re_call(self, x):
        """Avoid recomputing the square distances between x and itself, must be careful that you are using the same x repeatedly"""
        try:
            self.sq_dists
        except AttributeError:
            self.sq_dists = cdist(x, x, metric = 'sqeuclidean')
        return np.exp( - 0.5 * self.sq_dists / self.lengthscale)
