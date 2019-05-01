import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

class MDS:
    """ Classical multidimensional scaling (MDS)
                                                                                               
    Args:                                                                               
        D (np.ndarray): Symmetric distance matrix (n, n).          
        p (int): Number of desired dimensions (1<p<=n).
                                                                                               
    Returns:                                                                                 
        Y (np.ndarray): Configuration matrix (n, p). Each column represents a 
            dimension. Only the p dimensions corresponding to positive 
            eigenvalues of B are returned. Note that each dimension is 
            only determined up to an overall sign, corresponding to a 
            reflection.
        e (np.ndarray): Eigenvalues of B (p, ).                                                                     
                                                                                               
    """    
    @staticmethod
    def cmdscale(D, p = None):
        # Number of points                                                                        
        n = len(D)
        # Centering matrix                                                                        
        H = np.eye(n) - np.ones((n, n))/n
        # YY^T                                                                                    
        B = -H.dot(D**2).dot(H)/2
        # Diagonalize                                                                             
        evals, evecs = np.linalg.eigh(B)
        # Sort by eigenvalue in descending order                                                  
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        # Compute the coordinates using positive-eigenvalued components only                      
        w, = np.where(evals > 0)
        L  = np.diag(np.sqrt(evals[w]))
        V  = evecs[:,w]
        Y  = V.dot(L)   
        if p and Y.shape[1] >= p:
            return Y[:, :p], evals[:p]
        return Y, evals
    
class HC:
    """ Hierarchical Clustering (HC) """
    
    @staticmethod
    def linkage(matrix, method='average', optimal_ordering=True, **kwargs):
        return linkage(matrix, method=method, optimal_ordering=optimal_ordering,
                       **kwargs)
    
    @staticmethod
    def colorfunc(linkage, default_color, leaf_colors):
        link_cols = {}
        for i, idx12 in enumerate(linkage[:,:2].astype(int)):
              c1, c2 = (link_cols[idx] if idx > len(linkage)
                          else leaf_colors[idx] for idx in idx12)
              link_cols[i+1+len(linkage)] = c1 if c1 == c2 else default_color 
        return link_cols