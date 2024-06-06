from utils import *
import numpy as np

def apply_pca(data, val = None,m = None):
    # Center the training data
    centered_data,_ = center_data(data)

    # Compute the train covariance matrix and eigenvectors
    covariance_matrix = (centered_data @ centered_data.T)/float(data.shape[1])
    eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Reduce to m dimensions
    if m == None:
        P = eigenvectors[:,::-1]
    else:
        P = eigenvectors[:,::-1][:,0:m]
    
    if val is None:
        return P.T @ data
    
    # Apply pca to both train and val data
    return P.T @ data, P.T @ val
