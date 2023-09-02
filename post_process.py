import warnings
import numpy as np
from itertools import groupby
from tqdm.notebook import tqdm

def pmax(vector, value):
    """Returns the vector with each element less than value replaced by value"""
    vector_copy = vector.copy()
    vector_copy[vector_copy < value] = 0
    return vector_copy
    
def normalise_density(estimates, bin_size, tolerance = 1e-6, max_iters = 500):
    """
    Normalise the conditional density estimate via a binary search algorithm.
    Remove negative values and ensure area is close to 1.
    """
    # Remove negative values (and values very close to being negative)
    estimates_ = estimates.copy()
    estimates_[estimates_ < 0] = 0
    
    # Approximate the area of the density with rectangles
    area = bin_size * pmax(estimates_, 0).sum()
    
    if area == 0:
        #warnings.warn("Warning...........Density values all below zero, return uniform density")
        return np.ones(estimates.shape[0]) / ( estimates.shape[0] * bin_size )

    # If we have underestimated, scale the density up everywhere equally
    if area < 1:
        estimates_[estimates_ > 0] = estimates_[estimates_ > 0] / area

    # Retrieve the current upper, lower and middle density value
    upper = np.max(estimates_)
    lower = 0
    middle = ( upper + lower ) /2

    for i in range(max_iters):
        # Compute the area of the density, reduced by the middle density value, with negative elements removed
        area =  bin_size * np.sum( pmax(estimates_ - middle, 0) )
        # If the area is now within tolerance to 1, we are done
        if abs(area - 1) < tolerance:
            break
        # If we still need to scale down, set the new lower value to be the current middle value
        if area > 1:
            lower = middle
        # If we need to scale up, set the new upper value to be the current middle value
        else:
            upper = middle
        # Compute the new middle 
        middle = (upper + lower) / 2 

    if i == max_iters - 1:
        warnings.warn(f'Warning...........Reached maximum iterations, returning final area estimate = {area},\
                      consider increasing max_iters or decreasing tolerance.')

    # Return the normalised density estimates
    return pmax(estimates_ - middle, 0)

# This could be improved to also identify bumps that aren't zeroed on either side
def smooth(estimates, bin_size, delta):
    """
    Smooth the conditional density estimate by removing bumps that occur in between zero density regions
    by approximating the area of the bumps and setting the corresponding density estimates to zero if the
    area is below delta threshold.
    """
    # Copy to avoid in place assignment
    estimates_ = estimates.copy()

    # Run-length encode the vector to identify bumps which are zero on either side
    runs = np.array([ ( k, sum(1 for i in g) ) for k, g in groupby( estimates_ > 0 ) ])

    # Extract the values and the lengths of the runs
    values = runs[:, 0]
    lengths = runs[:, 1]
    n_runs = lengths.shape[0]

    # If the density estimate is all negative or all positive, there are no small bumps to consider
    if n_runs == 1:
        return estimates_

    # Grab the indices corresponding to the beginning and end of each run
    lower = np.append(0, np.cumsum(lengths))
    upper = np.cumsum(lengths)

    # Loop through each run
    for i in range(n_runs):
        # If the run corresponds to zeros, skip
        if not values[i]:
            next
        # Compute the area of just the bump, if its smaller than the threshold erase it
        area = bin_size * np.sum( estimates_[lower[i]:upper[i]] )
        if area < delta:
            estimates_[lower[i]:upper[i]] = 0

    return estimates_

def sharpen(estimates, bin_size, alpha):
    """Sharpen the conditional density estimate"""
    estimates_ = estimates ** alpha
    return estimates_

def post_process_loss(estimates, y_linspace, y):
    """
    Estimate the L2 loss with approximate integration and nearest neighbour expectation approximation.
    Allows us to optimise the post processing parameters.
    """
    # Approximate the double integral of the conditional destiny estimates squared 
    term1 = np.mean(np.trapz(estimates**2, y_linspace.flatten()))
    
    # Find the nearest neighbour on the grid to the current true y value
    nns = [np.argmin(np.abs(y_linspace - y[ii])) for ii in range(estimates.shape[0])]
    
    # Approximate the expectation
    term2 = np.mean(estimates[range(estimates.shape[0]), nns])
    return term1 - 2 * term2

def optimise_delta(estimates, y_linspace, y, deltas, opt_alpha = None, verbose = False):
    """Chooses the smooting parameter threshold which minimises the approximate loss"""
    # Retrieve binsize and normalise the estimates
    bin_size = y_linspace[1] - y_linspace[0]
    estimates_ = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size)
    
    # If we have already optimised a sharpening parameter, use it
    if opt_alpha is not None:
        estimates_ = np.apply_along_axis(sharpen, 1, estimates_, bin_size = bin_size, delta = opt_alpha) # Sharpen
        estimates_ = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size) # Re-normalise
            
    losses = np.zeros(deltas.shape[0])
    for i in tqdm(range(deltas.shape[0]), disable = not verbose):
        estimates = np.apply_along_axis(smooth, 1, estimates_, bin_size = bin_size, delta = deltas[i]) # Smooth each cde with choice of delta
        estimates = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size) # Re-normalise each cde
        losses[i] = post_process_loss(estimates, y_linspace, y)
    return losses, deltas[np.argmin(losses)]

def optimise_alpha(estimates, y_linspace, y, alphas, opt_delta = None, verbose = False):
    """Chooses the sharpening parameter  which minimises the approximate loss"""
    # Retrieve binsize and normalise the estimates
    bin_size = y_linspace[1] - y_linspace[0]
    estimates_ = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size)
    
    # If we have already optimised a smoothing parameter, use it
    if opt_delta is not None:
        estimates_ = np.apply_along_axis(smooth, 1, estimates_, bin_size = bin_size, delta = opt_delta) # Smooth
        estimates_ = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size) # Re-normalise
        
    losses = np.zeros(alphas.shape[0])
    for i in tqdm(range(alphas.shape[0]), disable = not verbose):
        estimates = np.apply_along_axis(sharpen, 1, estimates_, bin_size = bin_size, alpha = alphas[i]) # Sharpen each cde with choice of alpha
        estimates = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size) # Re-normalise each cde
        losses[i] = post_process_loss(estimates, y_linspace, y)
    return losses, alphas[np.argmin(losses)]

def optimise_delta_alpha(estimates, y_linspace, y, deltas, alphas, verbose = False):
    """Chooses both the sharpening and smoothing parameter which jointly minimises the approximate loss"""
    # Retrieve binsize and normalise the estimates
    bin_size = y_linspace[1] - y_linspace[0]
    estimates_ = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size)

    losses = np.zeros((deltas.shape[0], alphas.shape[0]))
    for i in tqdm(range(deltas.shape[0]), disable = not verbose):
        for j in range(alphas.shape[0]):
            estimates = np.apply_along_axis(smooth, 1, estimates_, bin_size = bin_size, delta = deltas[i]) # Smooth each cde with choice of delta
            estimates = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size)  # Re-normalise each cde
            estimates = np.apply_along_axis(sharpen, 1, estimates, bin_size = bin_size, alpha = alphas[j]) # Sharpen each cde with choice of alpha
            estimates = np.apply_along_axis(normalise_density, 1, estimates, bin_size = bin_size) # Re-normalise each cde
            losses[i, j] = post_process_loss(estimates, y_linspace, y)
            
    delta_ind, alpha_ind = np.where(losses == np.min(losses))
    return losses, deltas[delta_ind], alphas[alpha_ind]