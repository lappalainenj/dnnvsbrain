# -*- coding: utf-8 -*-
"""
The Activations class is the core of the library. BrainActivations and/ or
DNNActivations can be object of further investigation.

Todo:
    * Test Concat Class on all pretrained Pytorch Modules, if necessary extend
        a in method _get_module.
    * Make it free to choose whether to reassign layers or to provide model in
        a custom architecture.
"""

from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torchvision.models as models
from scipy.io import loadmat

from .datasets import Dummy


class Activations:

    """An abstract class of activations.
    
    Args:
        randomstate (int): Arbitrary integer to initialize a randomstate object.
    
    Attributes:
        sample (dict): A static dictionary that stores activation matrices
            per layer.
        activations (dict): A dynamic dictionary with a subsample of sample.
        layers (list): A list of the layers of interest.
        random (numpy.random.RandomState): A randomstate for reproducible sub-
            sample results.
    """
    def __init__(self, randomstate = 77, **kwargs):
        super(Activations, self).__init__()
        self.sample = {}
        self.activations = {}
        self.layers = []
        self.random = np.random.RandomState(randomstate)
    
    def __getitem__(self, layer):
        if isinstance(layer, str):
            return self.activations[layer]
        if layer > len(self.layers)-1:
            return IndexError, 'layer%s does not exist'%(layer+1)
        return self.activations['layer%s'%(layer+1)]  
    
    def __len__(self):
        return NotImplemented
    
    def fullsample(self):
        """Stores a deepcopy of the fullsample in activations."""
        self.activations = deepcopy(self.sample)
        
    def subsample_all(self, size, stochastic=False):
        """Stores subsamples of all activations in self.activations.
        
        Args:
            size (int): Desired size of the subsample.
            stochastic (bool): Whether to infer the probability for a certain
                sample point according to the magnitude of activation.
        """
        for layer in self.layers:
            self.subsample(layer, size, stochastic)

    def subsample(self, layer, size, stochastic=False):
        """Stores subsamples of the activations of layer in self.activations.
        
        Args:
            layer (str): Layer.
            size (int): Desired size of the subsample.
            stochastic (bool): Whether to infer the probability for a certain
                sample point according to the magnitude of activation.
        """
        if not size:
            return None
        else:
            subsample_ind = self._subsample_ind(layer, size, stochastic)  # get indices
            self.activations[layer] = self.sample[layer][:, subsample_ind]

    def _subsample_ind(self, layer, size, stochastic):
        """Creates random subsample indices per layer.
        
        Args:
            layer (str): Layer.
            size (int): Desired size of the subsample.
            stochastic (bool): Whether to infer the probability for a certain
                sample point according to the magnitude of activation.
        
        Returns:
            subsample_ind (array): Sample indices.
        """
        ind = np.arange(self.num_units[layer])
        try:
            p = None
            if stochastic:
                p = self._subsample_distribution(layer)
            subsample_ind = self.random.choice(ind, size=size, p=p,
                                                  replace=False)
        except Exception as e:
            print(('Samplesize higher than layer units for %s. '
                   'Taking all %s activations instead.' % (layer, ind.size)))
            subsample_ind = np.arange(0, ind.size, 1)
        return subsample_ind

    def _subsample_distribution(self, layer):
        """Gives a probability distribution to the subsampling routine 
        according to magnitude of activation."""
        values = self.activations[layer]
        p = np.abs(values).sum(axis=0) / np.abs(values).sum()
        return p
        
    def sort(self):
        """Sort activations according to number of elements for performance."""
        self.activations = {key:self.activations[key] # sort for performance
                            for key in sorted(self.activations,
                                key=lambda kv: self.activations[kv].size)}
    
    def _compute(self):
        return NotImplemented
    
    def dump(self, name=None):
        """Save activations to load them again later.
        
        Args:
            name (str): Filepath and name.
            
        Note: Appends a _temp to the name.
        
        Todo:
            *Generic location.
        """
        name += '_tmp'
        np.save(name, self.activations)
    
    def load(self, name=None):
        """Load activations into module.
        
        Args:
            name (str): Filepath and name.
            
        Note: Appends a _temp.npy to the name.
        
        Todo:
            *List names from generic location.
        """
        name += '_tmp.npy'
        self.sample = np.load(name).tolist()
        self.fullsample()
        
    def load_from_mat(self, filepath):
        """Load activations from matlab file into module.
        
        Args:
            filepath (str): Filepath and name.
        """
        data = loadmat(filepath)
        self.sample.update({layer: data[layer] for layer in self.layers})
        self.fullsample()
        
        
class BrainActivations(Activations):
    """Previously measured and preprocessed/ prepared brain activations.
   
    Args:
        filepath (pathlib.Path): The location of the data. The data is supposed
            to be a dictionary with brain regions as keys and (N, *) arrays as
            activation matrices, where N is the number of samples.
        layers (list of str): The recorded brain regions.
        mat (bool): Whether the data comes from a matlab structure.
            
    Attributes:
        layers (list of str): The recorded brain regions.
        layer_map (list of str): Maps the brain region names onto 'layer[i]'.
        num_units (dict): Number of elements of each layer activation.
        sum_units (dict): Total number of elements of all layer activations
            (specified in layers).
    """
    def __init__(self, filepath, layers=['AM', 'HC', 'EC', 'PHC'], mat=True):
        super(BrainActivations, self).__init__()
        self.layers = layers
        self.layer_map = {'layer%s'%(i+1):layer for i, layer in enumerate(self.layers)}
        if mat:
            self.load_from_mat(filepath)
        self.num_units = {k:v.shape[-1] for k, v in self.sample.items()}
        self.sum_units = sum([v for v in self.num_units.values()])

    class model:
        @staticmethod
        def _get_name():
            return 'Brain'
        
    def __len__(self):
        return len(self.sample[self.layers[0]]) # N, number of samples
    
    def __getitem__(self, layer):
        if isinstance(layer, str):
            return self.activations[layer]
        if layer > len(self.layers)-1:
            return IndexError, 'layer%s does not exist'%(layer+1)
        return self.activations[self.layer_map['layer%s'%(layer+1)]]
        
    
class DNNActivations(Activations):
    """Extract layer activations from a DNN according to images in a dataset.
   
    Args:
        model (str or torch.nn.module): String specifying pretrained model or
            a custom torch.nn.module
        dataset (torch.utils.data.Dataset): A dataset for which to compute the
            activations.
        pretrained (bool): In case of specifying a pretrained pytorch model.
            Default is True.
        layers (list of str): This list limits the layers for which to store the
            activations. E.g. ['layer1', 'layer2']. Default is all layers
        activation_functions (list of str): The module will be rearrange so that
            a layer is after any of these activation functions.
            Default is ['ReLU', 'Sigmoid']
            
    Attributes:
        model (torch.nn.module): A DNN as torch.nn.module.
        layers (list of str): This list limits the layers for which to store the
            activations. E.g. ['layer1', 'layer2']. Default is all layers
        num_parameters (dict): A dict containing parameters of the NN per layer.
        sum_parameters (int): All parameters in the NN.
        _shapes (dict): Shape of each layer activation.
        num_units (dict): Number of elements of each layer activation.
        sum_units (dict): Total number of elements of all layer activations
            (specified in layers).
    
    Note: Monkey patches the models forward function, so that the models output
        is a dictionary with activation per layer.
    """    
    def __init__(self, model, dataset = None, pretrained = True, layers = None,
                 activation_functions = ['ReLU', 'Sigmoid']):
        super(DNNActivations, self).__init__()
        
        if isinstance(model, str):
            try:
                self.model = eval('models.%s(pretrained = %s)'%(model, pretrained))
            except:
                pass
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            assert hasattr(self, 'model'), 'No model found.'
        
        reassign_layers(self.model, activation_functions) #TODO: Should not be obligatory
        self.layers = layers
        if not layers:
            self.layers = list(self.model._modules)
            
        self.num_parameters = {layer:sum([p.numel() for p in 
              self.model._modules[layer].parameters()]) for layer in self.layers}      
        self.sum_parameters = sum([v for v in self.num_parameters.values()])
        
        self.model.forward = _forward.__get__(self.model, self.model.__class__)
        self.model.extracted_layers = self.layers
        
        if dataset:
            if isinstance(dataset, torch.utils.data.Dataset):
                self.dataset = dataset
            if isinstance(dataset, dict):
                self.dataset = Dummy(**dataset)
            self._shapes = {k:v.shape for k, v in 
                            self.model(self.dataset[0][1].unsqueeze(0)).items()}
            self.num_units = {k:np.prod(v) for k, v in 
                                    self._shapes.items()}
            self.sum_units = sum([v for v in self.num_units.values()])
    
    def compute(self):
        """Tests the availability of memory to store the activations within a 
        numpy array. If enough Memory is available it calls _compute."""
        try:
            np.zeros([2, len(self), self.sum_units])
            self._compute()
        except MemoryError:
            raise MemoryError(('Not enough memory to store activations of %s layers'
                  ' with %s units of %s images. Reduce dataset size'
                  ' or select a subset of layers (default is all).')
                    %(len(self.layers), self.sum_units, len(self)))
        except IndexError as e:
            raise IndexError(e, 'Check your dataset configuration.')              
    
    def _compute(self):
        """Computes the activations."""
        self.sample = {key: np.zeros([len(self),
                                      self.num_units[key]])
                            for key in self.layers}
    
        self.model.eval()
        # layer activations for all images
        for i, (_, image) in enumerate(tqdm(self.dataset,
                                       desc = 'Computing activations')):
            out = self.model(image.unsqueeze(0))
            for layer in self.layers:
                self.sample[layer][i] = out[layer].detach().numpy().flatten()
        self.fullsample() # deepcopy sample into activations
        
    def __len__(self):
        return len(self.dataset)
        
                    
def reassign_layers(model, last_function = ['ReLU', 'Sigmoid']):
    """Takes each module in a model and restructures the functions into modules
    called layer[i] according to the position i of a function specified in 
    last_function. E.g. one would like to have layers always end with an
    activation function, then last_function should be filled with all names of
    activation functions in the model.
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        last_function (list of str): Function names that are used to identify 
            the end of a layer.
    """
    Concat(model)
    module = model._modules.pop('concatenated')
    functions, layers = _get_functions(module, last_function) # 
    layers = _sort_layers(module, functions, layers)
    _fill_modules(model, layers)
    

class Concat():
    """Concats all modules of the model.
    
    Note: Be cautious with recursive/residual networks. Certain blocks that 
        define the layer architecture by the forward function should stay intact.
    
    Todo:
        * Test Concat Class on all pretrained Pytorch Modules, if necessary extend
        a in method _get_module.
    
    Args:
        model (torch.nn.Module): The module which submodules will
            be concatenated to a single module called concatenated.
    
    Attributes:
        model (torch.nn.Module): The module which submodules will
            be concatenated to a single module called concatenated.
        list_of_layers (list): List to temporally store the submodules of the 
            network.
    """
    def __init__(self, model):
        self.model = model
        self.list_of_layers = []
        self._get_modules(model)
        self._concat()
    
    def _get_modules(self, model):
        """Recursively puts all submodules into list_of_layers.
        Note: Submodules in residual/recursive networks should not be split!"""
        from sys import modules
        from inspect import getmembers, isclass
        #enable for resnet, tricky: shallow and nested nn modules / keep BasicBlocks
        a = [x[1] for x in getmembers(modules[models.resnet.__name__], isclass)]
        for module in model.children():
            if not list(module.children()) or isinstance(module, tuple(a)):
                self.list_of_layers.append(module)
            else:
                self._get_modules(module)
        
    def _concat(self):
        modules = [module for module in self.model._modules]
        for module in modules:
            self.model._modules.pop(module)
        self.model._modules['concatenated'] = nn.Sequential(*self.list_of_layers)


def _get_functions(module, last_function):
    """Identifies the positions i of the functions, specified in last_function,
    and thus defining the end of the new layers.
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        last_function (list of str): Function names that are used to identify 
            the end of a layer.
    
    Returns:
        functions (list): A list containing the keys of all functions that were
            identified in the submodule.
        layers (dict): A dict containing a key, value pair of a layer index and
            an empty list.
    """
    functions = []
    layers = {}
    lastsubmodule = False
    for key, submodule in module._modules.items():
        if key == next(reversed(module._modules)):
            lastsubmodule = True
        match = [function in submodule.__str__() for function in last_function]
        if any(match+[lastsubmodule]):
            functions.append(key)
    layers.update({i+1:[] for i in range(len(functions))})
    return functions, layers


def _sort_layers(module, functions, layers):
    """Constructs the new layers. The modules are filled into the lists that
    describe the new layers in ascending order up to the final function. 
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        functions (list): A list containing the keys of all functions that were
            identified in the submodule.
        layers (dict): A dict containing a key, value pair of a layer index and
            an empty list.
    
    Returns:
        layers (dict): A dict containing a key, value pair of a layer index and 
            a list with all respective torch.nn.functions.
    """
    for i, key in enumerate(functions):
        for j in range(int(key)+1):
            try:
                layers[i+1].append(module._modules.pop(str(j)))                
            except:
                pass
    return layers


def _fill_modules(model, layers):
    """Creates the new nn.modules according to the extracted layers and assigns
    them to the model. 
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        layers (dict): A dict containing a key, value pair of a layer index and 
            a list with all respective torch.nn.functions.
    """
    for key, value in layers.items():
        model._modules['layer%s'%(key)] = nn.Sequential(*value)


def _forward(self, x):
    """A generic forward function, which stores the activations specified in 
    extracted_layers. It works for pure forward networks including transitions
    of convolution and fully connected layers. 
    
    Note: Tested with AlexNet, VGG. ResNet with its BasicBlocks kept intact.
    """
    outputs = {}
    for name, module in self._modules.items():
        for subname, submodule in module._modules.items():
            try:
                x = submodule(x)
            except:
                x = x.view(x.size(0), -1)
                x = submodule(x)
        if self.extracted_layers and name in self.extracted_layers:
            outputs[name] = x
            if name == self.extracted_layers[-1]:
                return outputs
    if outputs:
        return outputs
    return x
