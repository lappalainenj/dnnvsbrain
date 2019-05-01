from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import matplotlib.pylab as plt
from torchvision import transforms

from .activations import Activations

class Dimensionality:
    """Extract Image and Layer Statistics"""

    def __init__(self):
        self.explained_variance = {}
        self.explained_variance_fit = {}
        self.alpha = {}
        self.X = []

    def __call__(self, activations, regmethod=Ridge(alpha=1.0)):
        self.activations = activations
        self.dataset = self.activations.dataset
        self.layers = self.activations.layers
        self.dimensions = len(self.activations)
        self.X = np.arange(1, self.dimensions).reshape(-1, 1)
        self.compute(regmethod)

    def compute(self, regmethod=Ridge(alpha=1.0)):
        """Computes the explained variance per dimensions per layer/region and
        a linear fit which is giving alpha."""
        for layer in self.layers:
            X, y = self.pca(self.activations[layer])
            # y_pred, alpha = self.regression(X, y, regmethod=regmethod)
            self.explained_variance[layer] = y
            #self.explained_variance_fit[layer] = y_pred
            #self.alpha[layer] = alpha
            
    def fit(self, alpha):
        """Fits the power law to the explained variance using regression"""
        for layer in self.explained_variance:
            y_pred, alpha = self.regression(self.X,
                                            self.explained_variance[layer],
                                            regmethod = Ridge(alpha=alpha))
            self.explained_variance_fit[layer] = y_pred
            self.alpha[layer] = alpha
            
    def omitlast(self, n):
        for layer in self.explained_variance:
            self.explained_variance[layer] = self.explained_variance[layer][:-n]
        self.X = self.X[:-n]
            
    @staticmethod
    def pca(activation):
        """PCA for activation (N, Units)."""
        pca = PCA()
        pca.fit(activation)
        X = np.arange(1, pca.n_samples_).reshape(-1, 1)
        y = pca.explained_variance_ratio_[:-1].reshape(-1, 1)
        return X, y
    
    @staticmethod
    # TODO: fit power law correctly
    def regression(X, y, regmethod=LinearRegression()):
        """Loglog linear regression of explained variance."""
        regmethod.fit(np.log10(X), np.log10(y))
        y_pred = X**regmethod.coef_*10**regmethod.intercept_
        alpha = -1*regmethod.coef_[0][0]
        return y_pred, alpha

    @staticmethod
    def get_powerlaw(y, x):
        allrange = np.arange(y.size)
        logy = np.log(np.abs(y))
        Y = logy[x][:, None]
        X = -np.log(x + 1).T[:, None]
        w = 1. / (x + 1)[:, None]
        X = np.hstack((X, np.ones([X.size, 1])))
        B = np.linalg.inv(X.T @ (w * X)) @ ((w * X).T @ Y)
        lpts = np.round(np.exp(np.linspace(np.log((x + 1)[0]), np.log((x + 1)[-1]), 100))).astype(int);
        r = np.corrcoef(np.log(lpts), logy[lpts - 1])[0, 1]
        r = r ** 2
        X = np.hstack((-np.log(allrange + 1)[:, None], np.ones([allrange.size, 1])))
        ypred = np.exp(X @ B).flatten()
        alpha = B[0][0]
        b = B[1][0]
        return ypred, alpha, b, r
    
    def dataset_stat(self, transform_off=True):
        """Get statistics of the images."""
        if transform_off:
            x = self.dataset.transform
            self.dataset.transform = transforms.Compose([])
        self.dataset.add_transform(np.array)
        X, y = self.pca([img[1].flatten() for img in self.dataset])
        y_pred, alpha = self.regression(X, y)
        if transform_off:
            self.dataset.transform = x
        self.explained_variance['imageset'] = y
        self.explained_variance_fit['imageset'] = y_pred
        self.alpha['imageset'] = alpha
        return X, y, y_pred, alpha
    
    def _ev_lambda(self, pipeline):
        '''Used for obtaining plot limits via min, max function.'''
        ev = [self.explained_variance[key] for key in self.explained_variance]
        for f in pipeline:
            ev = f(ev)
        return ev
    
    # def __getattr__(self, attr):
    #     return getattr(self.activations, attr)
    #
    def dump(self, name=None):
        """Save calculated value to load them again later.
        
        Args:
            name (str): Filepath and name.
            
        """
        data = {k:self.__dict__[k] for k in ('X',
                                            'explained_variance',
                                            'explained_variance_fit',
                                            'alpha')}
        np.save(name, data)
    
    def load(self, name=None):
        """Load data into module.
        
        Args:
            name (str): Filepath and name.
            
        """
        data = np.load(name).tolist()
        self.X = data['X']
        self.explained_variance = data['explained_variance']
        self.explained_variance_fit = data['explained_variance_fit']
        self.alpha = data['alpha']

    def append(self, attribute, key, value):
        getattr(self, attribute).update({key: value})
    
def whiten(X, fudge=1E-18):

       # the matrix X should be samples - pixels
       X = X.T
       # get the covariance matrix
       Xcov = np.dot(X.T,X)

       # eigenvalue decomposition of the covariance matrix
       d, V = np.linalg.eigh(Xcov) # V is samples x samples

       # a fudge factor can be used so that eigenvectors associated with
       # small eigenvalues do not get overamplified.
       D = np.diag(1. / np.sqrt(d+fudge))

       # whitening matrix
       W = np.dot(np.dot(V, D), V.T) # samples x samples

       # multiply by the whitening matrix
       X_white = np.dot(X, W)

       return X_white.T