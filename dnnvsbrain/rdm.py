import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import kendalltau

class RDM:
    """Initializes an abstract RDM object for an activations object.
   
    Args:
        activation (Activation): The computed activations.
        subsamplesize (int): Activation units to consider.
        stochastic (bool): Wether to use stochastic weighting of units.
        repetitions (int): Number of subsampling from the activations.
            Default is 1.
    """    
    
    def __init__(self, activations, subsamplesize=None,
                 stochastic=False,repetitions=1, **kwargs):       
        super(RDM, self).__init__()
        self.activations = activations
        self.subsamplesize = subsamplesize
        self.stochastic = stochastic
        self.repetitions = repetitions
        self.rdm = {}
        
    def __getitem__(self, layer):
        # string indexing
        if isinstance(layer, str):
            return self.rdm[layer]
        # integer indexing
        if layer > len(self.layers)-1:
            return IndexError, 'layer%s does not exist'%(layer+1)
        return self.rdm['layer%s'%(layer+1)]

    def __getattr__(self, attr):
        return getattr(self.activations, attr)     
        
    def compute(self):
        """Computes the rdms."""
        self.rdm.update({key:np.zeros([self.repetitions,
                                       len(self.activations),
                                       len(self.activations)]) for key in self.layers})
        for layer in self.layers:
            for r in range(self.repetitions):
                self.subsample(layer, self.subsamplesize, self.stochastic)    
                self.rdm[layer][r] = self.get_rdm(self.activations[layer])
    
    @staticmethod
    def get_rdm(activation):
        """Calculates the distance of the activations.
        
        Args:
            activations (np.ndarray): List of activation observations of the 
                layer to the individual stimuli. 

        Returns:
            np.ndarray: similarity matrix
        """
        result = squareform(pdist(activation, metric = 'correlation'))
        return result
    
    def mean_rdm(self, layer):
        """Averages the rdm matrices over repetitions."""
        return self.rdm[layer].mean(axis = 0)
        
class Correlation:
    """Handles correlations of two rdm matrix objects. A mask can be defined
    generically to define certain regions of interest."""
   
    def __init__(self, rdm1, rdm2, experiment, method=kendalltau):
        super(Correlation, self).__init__()
        assert len(rdm1.activations)==len(rdm2.activations), ("Activations do"
                                              " not underlie the same Dataset"
                                              " of equal length.")
        self.len = len(rdm1.activations)
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.repetitions1 = len(rdm1[rdm1.layers[0]])
        self.repetitions2 = len(rdm2[rdm2.layers[0]])
        self.layers1 = rdm1.layers
        self.layers2 = rdm2.layers
        self.method = method
        self.mask = experiment.mask
        self.correlation = {}
        self.indices = np.zeros((self.len, self.len)).astype('bool')
        self.blocks = {}
        self.colordict = experiment.correlation_colors

        
    def __getitem__(self, roi):
        return self.correlation[roi]
        
    def compute(self):
        self._blocks(self.mask)
        for mask in self.blocks:
            self.correlation.update({mask:{}})
            for layer1 in self.layers1:
                self.correlation[mask].update({layer1:{}})
                for layer2 in self.layers2:
                    self.correlation[mask][layer1].update({layer2:{}})
                    corr_matrix = self.stack(layer1, layer2, mask)
                    self.correlation[mask][layer1][layer2]['mean']=corr_matrix.mean()
                    self.correlation[mask][layer1][layer2]['std']=corr_matrix.std()
      
    def stack(self, layer1, layer2, mask):
        stacked = np.zeros([self.repetitions1+self.repetitions2,
                            self.blocks[mask].sum()])
        for r in range(self.repetitions1):
            stacked[r] = self.rdm1[layer1][r][self.blocks[mask]].flatten()
        for r in range(self.repetitions2):
            stacked[self.repetitions1 + r] = self.rdm2[layer2][r][self.blocks[mask]].flatten()
        return squareform(pdist(stacked, metric = self._method))[:self.repetitions1,self.repetitions1::]
    
    def _method(self, *args, **kwargs):
        return self.method(*args, **kwargs)[0]
        
    def _corr_f(self, f):
        '''Used for obtaining plot limits via min, max function.'''
        cvalues = []
        for typ in self.correlation:
            for layer in self.correlation[typ]:
                for value in self.correlation[typ][layer].values():
                    cvalues.append(value['mean'])
        return f(cvalues)
        
    def _blocks(self, mask, within = False):
        """Computes a mask per region of interest defined in self.mask"""
        for typ in mask:
            if typ == 'within':
                # recursive call to get mask for roi that belongs to within
                self._blocks(mask[typ], within = True)
                # get mask for all within rois
                indices = np.zeros((self.len, self.len)).astype('bool')
                for key in mask[typ]:
                    r1, r2, c1, c2 = mask[typ][key]
                    triag = np.triu_indices(n = r2-r1, m = c2-c1, k = 1)
                    indices[r1:r2, c1:c2][triag] = True
                self.blocks[typ] = indices 
            else:
                indices = np.zeros((self.len, self.len)).astype('bool')
                r1, r2, c1, c2 = mask[typ]
                if within:
                    # get mask for roi that belongs to within
                    triag = np.triu_indices(n = r2-r1, m = c2-c1, k = 1)
                    indices[r1:r2, c1:c2][triag] = True
                else:
                    # get mask for everything else
                    indices[r1:r2, c1:c2] = True
                self.blocks[typ] = indices