"""Data utility functions.
TODO: Update docstrings.
"""
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import skimage.io as io
import matplotlib.pylab as plt


class TransformDataset:

    def __init__(self, transform=[], normalize=[], experiment = None):
        """
        Initialize transformation.
        """
        super(TransformDataset, self).__init__()

        if transform:
            if transform == 'default':
                self.transform = transforms.Compose([transforms.Resize(256)
                                                        , transforms.CenterCrop(224)
                                                        , transforms.ToTensor()])
            else:
                self.transform = transform
        else:
            self.transform = transforms.Compose([])

        if normalize:
            if normalize == 'default':
                default_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                self.transform.transforms.append(default_norm)
            else:
                self.transform.transforms.append(normalize)
        else:
            pass
        self.experiment = experiment
        self._plot_examples = False
    
    def add_norm(self, mean, std):
        norm = transforms.Normalize(mean=mean, std=std)
        self.add_transform(norm)
    
    def add_transform(self, transform):
        self.transform.transforms.append(transform)
        
    def default_transform(self):
        self.transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])
    
    def default_norm(self):
        default_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.add_transform(default_norm)
        
    def __getattr__(self, attr):
        return getattr(self.experiment, attr)     

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item(key)
        else:
            raise TypeError("Invalid argument type.")
            
    def plot_examples(self, randomstate=77, transform_off=True,
                      title = False, show = True, save = False):
        self._plot_examples = True
        rs = np.random.RandomState(randomstate)
        indices = np.arange(0, len(self))
        indices = rs.choice(indices, 16, replace=False).astype('int')
        fig, ax = plt.subplots(4, 4, figsize = [12, 12])
        if transform_off:
            x=self.transform
            self.transform = transforms.Compose([transforms.Resize(256)
                                            ,transforms.CenterCrop(224)])
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                img = self[int(indices[i+j*4])][1]
                if isinstance(img, torch.Tensor):
                    img = img.numpy().transpose((1, 2, 0))
                ax[i, j].imshow(img)
                ax[i, j].set_axis_off()
        plt.subplots_adjust(wspace=0.1, hspace=0)
        if title:
            plt.suptitle(title)                
        if save:
            file_name = input("Filename? ...")
            plt.savefig(self.fig_path / file_name,
                        dpi = 300, bbox_inches='tight')
        if show:
            plt.show()
        if transform_off:
            self.transform=x
        self._plot_examples = False
        
    def plot_all(self, transform_off=True, title = False, show = True, save = False):
        n = int(np.floor(np.sqrt(len(self))))
        fig, ax = plt.subplots(n, n, figsize = [12, 12])
        if transform_off:
            x=self.transform
            self.transform = transforms.Compose([transforms.Resize(256)
                                            ,transforms.CenterCrop(224)])
        for i in range(n):
            for j in range(n):
                img = self[int(i+j*n)][1]
                if isinstance(img, torch.Tensor):
                    img = img.numpy().transpose((1, 2, 0))
                ax[i, j].imshow(img)
                ax[i, j].set_axis_off()
        plt.subplots_adjust(wspace=0.1, hspace=0)
        if title:
            plt.suptitle(title)                
        if save:
            plt.savefig(self.fig_path / save,
                        dpi = 300, bbox_inches='tight')
        if show:
            plt.show()
        if transform_off:
            self.transform=x
        self._plot_examples = False


class RDMDataset(TransformDataset, Dataset):

    
    def __init__(self, image_names=None, image_path=None, transform=None,
                 normalize=None, target=False, shuffle=False, state=None,
                 test_separate=None, experiment = None, fig_path = None):
        """
        Initialize Dataloader.
    
        Parameters
        ----------
        transform : object of torchvision.transforms.Compose  
        """
        assert any([image_names, hasattr(experiment, 'image_names')]), ("Please specify"
                   " either experiment.image_names or image_names.")
        assert any([image_path, hasattr(experiment, 'image_path')]), ("Please specify"
                   " either experiment.image_path or image_path.")
        super(RDMDataset, self).__init__(transform=transform,
                                         normalize=normalize,
                                         experiment=experiment)
        image_names = image_names if image_names else experiment.image_names            
        self.image_names = ImageNames(test_separate, state, image_names)
        if shuffle:
            self.image_names.shuffle(state)
        self.image_path = image_path if image_path else experiment.image_path
        self.fig_path = fig_path if fig_path else experiment.fig_path
        self.target = target
#        self.target_category = experiment.target_category
#        self.category_numer = experiment.category_number
        
    def __len__(self):
        return len(self.image_names)
            
    def __getattr__(self, attr):
            return getattr(self.image_names, attr)
           
    def get_item(self, idx):
        
        image = Image.open(self.image_path / self.image_names[idx])
        if self.transform:
            image = self.transform(image)
        #image.unsqueeze_(0)
        if self.target:
            target = self.get_target(self.image_names[idx])
            return target, image
        return self.image_names[idx], image

    def get_target(self, img_name):
        idx = int(self.category_number[self.target_category[img_name]])
        return idx#torch.Tensor([idx])
        #target = torch.zeros(10)
        #target[idx] = 1
        #return target
            
class ImageNames():
    
    def __init__(self, test_separate = 0, state = None, image_names = None):
        self.image_names = deepcopy(image_names)
        self.backup = deepcopy(image_names)
        self._test = False
        if test_separate:
            self.shuffle(state)
            self.separate=deepcopy(self.image_names[0:int(len(image_names)*test_separate)])
            self.image_names=self.image_names[int(len(image_names)*test_separate):]
        
    def __getitem__(self, idx):           
        if self._test:
            return self.separate[idx]
        return self.image_names[idx]
        
    
    def __len__(self):
        return len(self.image_names)
    
    def shuffle(self, state = None):
        if state:
            randomstate = np.random.RandomState(state)
            randomstate.shuffle(self.image_names)
        else:
            np.random.shuffle(self.image_names)
            
    def order(self):
        self.image_names = deepcopy(self.backup)
        
    def test(self):
        self._test=True
    
    def train(self):
        self._test=False
        
    
class CrossValDataset(Dataset):
    
    def __init__(self, full_ds, offset, length):
        super(CrossValDataset, self).__init__()
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough."
                                                      "Offset %s + Length %s "%(offset, length))

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if self.full_ds._test:
            return self.full_ds[i]
        return self.full_ds[i+self.offset]
    
def split_train_val(dataset, val_share=0.2):
    val_offset = int(len(dataset)*(1-val_share))
    return CrossValDataset(dataset, 0, val_offset), \
            CrossValDataset(dataset, val_offset, len(dataset)-val_offset)


def split_train_val_test(dataset, val_share=0.2, test_share=0.2):
    val_offset = int(len(dataset)*(1-(val_share+test_share)))
    test_offset = val_offset + int(len(dataset)*val_share)
    return CrossValDataset(dataset, 0, val_offset), \
            CrossValDataset(dataset, val_offset, test_offset-val_offset), \
            CrossValDataset(dataset, test_offset, len(dataset)-test_offset)


class Baseline(Dataset):

    def __init__(self, num_obs = 100, dims = [3, 224, 224]):
        super(Baseline, self).__init__()
        self.greyscale_values = np.linspace(0, 1, num_obs)
        self.num_obs = num_obs
        self.dims = dims
        
    def __len__(self):
        return self.num_obs
        
    def __getitem__(self, key):
        return self.greyscale_values[key], torch.ones(self.dims)*self.greyscale_values[key]


class Dummy(TransformDataset, Dataset):

    def __init__(self, data, transform = None, normalize = None):
        '''Data must be [N, *]'''
        super(Dummy, self).__init__(transform=transform, normalize=normalize)
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        if self.transform:
            return key, self.transform(self.data[key])
        return key, self.data[key]


class CocoTransform(TransformDataset, Dataset):

    def __init__(self, experiment=None, white=False, content_thresh=0.1,
                       transform = 'default', normalize = 'default',
                       randomstate = 77, categories = None, num_images = None,
                       annFile = None, debug_dataset = False, cocodir = None):
        from pycocotools.coco import COCO # cocoapi is a bit tricky to install
                                          # shouldnt be a requirement
        assert any([categories, hasattr(experiment, 'labels')]), ("Please specify"
                   " either experiment.labels or categories.")
        assert any([num_images, hasattr(experiment, 'labels')]), ("Please specify"
                   " either experiment.num_images or num_images.")
        assert any([cocodir, hasattr(experiment, 'cocodir')]), ("Please specify"
                   " either experiment.cocodir or cocodir.")
        assert any([annFile, hasattr(experiment, 'image_path')]), ("Please specify"
                   " either experiment.image_path or annFile.")
        super(CocoTransform, self).__init__(transform=transform,
                                         normalize=normalize,
                                         experiment=experiment)
        self.coco = COCO(annFile) if annFile else COCO(experiment.image_path)
        self.categories = categories if categories else experiment.labels
        self.num_images = num_images if num_images else experiment.num_images_per_label
        self.cocodir = cocodir if cocodir else experiment.cocodir
        self.white = white
        self.category_ids = {}
        self.content_thresh = content_thresh
        for category in self.categories:
            cat_ids = self.coco.getCatIds(catNms=[category])
            assert len(cat_ids)!=0, "Check category definitions. Maybe a typo?"
            img_ids = self.coco.getImgIds(catIds=cat_ids)
            self.category_ids[category] = img_ids
        self.indices = {}
        self.key2category = {}
        for i in range(len(self.categories)*self.num_images):
            self.key2category[i] = self.categories[i // self.num_images]
        self.random = np.random.RandomState(randomstate)
        self._category_ids = deepcopy(self.category_ids)
        self.debug_dataset = debug_dataset
        self.returned = {cat:[] for cat in self.categories}
            
    def __len__(self):
        return len(self.categories)*self.num_images
            
    def get_item(self, key):
        category = self.key2category[key] 
        img = self.image(category)
        img = Image.fromarray(img)
        if self.transform:
            return [category, key], self.transform(img)
        return [category,key], img

    def image(self, category):

        if len(self._category_ids[category]) == 0:
            raise AssertionError("The dataset does not meet the criteria. Add more"
                                 " data or decrease the content threshold.") #TODO: not helpful if _category_ids is empty
        img_id_idx = self.random.randint(len(self._category_ids[category]))
        img_id = int(self._category_ids[category][img_id_idx])
        img_meta = self.coco.loadImgs(img_id)[0]
        img = io.imread(self.cocodir / img_meta['file_name'])    #load image

        mask = self.mask(category, img_id)                  #get mask
        img_white = np.copy(img)
        img_white[np.where(mask == 0)] = 255
        
        # call recursively if image contains too small non-white content 
        if img_white[img_white!=255].size/img_white.size < self.content_thresh:
            self._category_ids[category].pop(img_id_idx)
            return self.image(category)
        if not self._plot_examples:
            self._category_ids[category].pop(img_id_idx) # returned[category] = img_id
        if self.debug_dataset:
            good = self.debug(category, img_id, img, img_white)
            if not good:
                return self.image(category)
        if self.white:
            return self.center(img_white)
        return img
    
    def debug(self, category, img_id, img, img_white):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(img_white)
        plt.show()
        user_in = input("Image ok for category %s? (y/n)"%category)
        if not any(('n' == user_in, 'y' == user_in)):
            self.debug(category, img_id)
        elif user_in == 'y':
            self.returned[category].append(img_id)
            return True
        else:
            return False
        
    def center(self, img_white):
        from scipy.ndimage.measurements import center_of_mass
        from skimage.transform import warp, AffineTransform
        H0, W0 = img_white.shape[0:2]
        H0 = int(H0 / 2)
        W0 = int(W0 / 2)
        com = center_of_mass(img_white - 255)
        H, W = np.round(com).astype('int')[0:2]
        at = AffineTransform(translation=[W0 - W, H0 - H])
        return (255*warp(img_white, at.inverse, cval=1.0)).astype('uint8')

    def mask(self, category, img_id):
        cat_ids = self.coco.getCatIds(catNms=[category])
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)  # annotations
        mask = self.coco.annToMask(anns[0])
        return mask