import numpy as np

import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image

from .experiment import ImagenetExperiment
from .datasets import RDMDataset


class Transform:
    
    def __init__(self):
        super(Transform, self).__init__()
        self.p = 1
        self.tensor=False
        self.pil=True
    
    def objectmask(self, img, threshold = 5):    
        def dist(channel, color):
            return np.sqrt((channel-color)**2) < threshold
        img = np.array(img)
        color = 255. # white
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        mask = dist(R,color)*dist(G,color)*dist(B,color)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
        return mask
    
    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        img = np.array(img)
        self.size = img.shape
        mask = self.objectmask(img)
        background = self.sample_bg(img, mask)
        img[mask] = background[mask]
        img = img.astype('uint8')
        if self.pil:
            return Image.fromarray(img)
        if self.tensor:
            return transforms.Tensor()(img)            
        return img
    

class WhiteToMean(Transform):
    
    def __init__(self, p = 1):
        super(WhiteToMean, self).__init__()
        
    def sample_bg(self, img, mask, *args, **kwargs):
        background = np.zeros(self.size).astype('uint8')
        mask = mask[...,0]
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        meanR, meanG, meanB = np.mean(R[~mask]), np.mean(G[~mask]), np.mean(B[~mask])
        background[..., 0] = int(meanR)
        background[..., 1] = int(meanG)
        background[..., 2] = int(meanB)
        return background
    
    
class NaturalBackground(Transform):
    """Input: numpy array or PIL image
    Output: pil, numpy array or tensor with natural background"""
    
    def __init__(self, pil=False, tensor=False, p = 1, threshold=5):
        super(NaturalBackground, self).__init__()
        experiment = ImagenetExperiment()
        data = RDMDataset(experiment=experiment)
        data.add_transform(np.array)        
        self.images = [img for _, img in data]
        self.tensor=tensor
        self.pil=pil
        self.threshold=5
        self.calc_hist()
        self.p = p
    
    def calc_hist(self):
        import cv2
        mask, histSize, ranges = None, [256], [0, 256]
        rhist = cv2.calcHist(self.images, [0], mask, histSize, ranges)
        self.rhist = rhist.flatten()/rhist.sum()
        ghist = cv2.calcHist(self.images, [1], mask, histSize, ranges)
        self.ghist = ghist.flatten()/ghist.sum()
        bhist = cv2.calcHist(self.images, [2], mask, histSize, ranges)
        self.bhist = bhist.flatten()/bhist.sum()
        
    def sample_bg(self, *args, **kwargs):
        background = np.zeros(self.size).astype('uint8')
        range = np.arange(0, 256)
        background[..., 0] = np.random.choice(range, size = self.size[:-1], p = self.rhist)
        background[..., 1] = np.random.choice(range, size = self.size[:-1], p = self.ghist)
        background[..., 2] = np.random.choice(range, size = self.size[:-1], p = self.bhist)
        return background
    
class PatchBackground():
    
    def __init__(self):
        experiment=ImagenetExperiment()
        data = RDMDataset(experiment=experiment)
        data.add_transform(transforms.Resize(28))
        data.add_transform(transforms.CenterCrop(28))
        data.add_transform(transforms.ToTensor())
        self.all_images = [img for _, img in data]
        
    def create_grid(self):
        background = torch.zeros([3, 224, 224])
        for i in range(8):
            for j in range(8):
                background[:,
                           int(224/8*i):int(224/8*(i+1)),
                           int(224/8*j):int(224/8*(j+1))]=self.all_images[i*8+j]
        return background
           
    def patchbackground(self, tensor, threshold = 0.05):
        np.random.shuffle(self.all_images)
        background = self.create_grid().numpy()
        def dist(a, b):
            return np.sqrt((a-b)**2) < threshold
        color = 1.0
        img = tensor.numpy()
        R, G, B = img
        mask = dist(R,color)*dist(G,color)*dist(B,color)
        mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
        img[mask] = background[mask]
        return Tensor(img)
    
    def __call__(self, tensor):
        return self.patchbackground(tensor)

def ColorToNan(tensor, color = [1.0, 1.0, 1.0], threshold = 0.05):
    def dist(a, b):
        return np.sqrt((a-b)**2) < threshold
    img = tensor.numpy() # C, W, H
    R, G, B = img
    mask = dist(color[0], R)*dist(color[1], G)*dist(color[2], B) # np.all(img == color, axis = -1)
    mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    img[mask] = np.nan
    return Tensor(img) # .transpose((2, 0, 1)))       

def WhiteToMean_b(tensor, threshold = 0.05):
    def dist(a, b):
        return np.sqrt((a-b)**2) < threshold
    color = 1.0
    img = tensor.numpy()
    R, G, B = img
    mask = dist(R,color)*dist(G,color)*dist(B,color)
    meanR, meanG, meanB = np.mean(R[~mask]), np.mean(G[~mask]), np.mean(B[~mask])
    img[0][mask] = meanR
    img[1][mask] = meanG
    img[2][mask] = meanB
    return Tensor(img)

class Transpose:
    def __init__(self):
        pass
    def __call__(self, img):
        return img.transpose((1, 2, 0))