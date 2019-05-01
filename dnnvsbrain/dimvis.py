import numpy as np
import matplotlib.pylab as plt
import re

from .experiment import fig_path

path = fig_path / 'dim'


class DimVis():
    
    def __init__(self, experiment=None, **kwargs):
        super(DimVis, self).__init__()
        self.experiment = experiment
        
    def plot(self, dim, exclude_layers = [], show = False, title = False,
             save = False, label = 'xt', thresh = 0.99):
        num_columns = len(dim.explained_variance)
        fig, ax = plt.subplots(1, num_columns, figsize = [18, 2])
#        ymin = dim._ev_lambda([np.min])
#        ymax = dim._ev_lambda([np.max])
        layers = [l for l in dim.explained_variance if l not in exclude_layers]
        layers.sort()
        for i, layer in enumerate(layers):
            #ax[i].set_ylim([ymin, ymax+0.05])
            idx = np.cumsum(dim.explained_variance[layer]) < thresh
            ax[i].loglog(dim.X[idx], dim.explained_variance[layer][idx])
            ax[i].loglog(dim.X[idx], dim.explained_variance_fit[layer][idx])
            if i != 0:
                ax[i].set_yticks([])
            if 'x' in label:
                ax[i].set_xlabel('PC dimension')
            if 't' in label:
                ax[i].set_title(re.sub('\d+',' \g<0>', layer).capitalize())
            ax[i].text(0.3, 0.9, 'alpha = %.4f'%(dim.alpha[layer]),
                                                  transform = ax[i].transAxes)
        if title:
            fig.suptitle(title, y = 1.1)
        if save:
            #if hasattr(dim.activations, 'model'): # not true for brainactivations
            plt.savefig(self.path / ('%s.png'%(save)),
                            dpi = 300, bbox_inches='tight')
#            else:
#                filename = input(('Saving figure in %s. Please insert a filename: ')
#                                 %path)
#                plt.savefig(path, filename,
#                            dpi = 300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod       
    def plot_imgstat(X, y, y_pred, alpha, label = 'xt', dim=None, *args):
        ax = plt.gca()
        ax.loglog(X, y)
        ax.loglog(X, y_pred)
        ax.text(0.3, 0.9, 'alpha = %.4f'%(alpha),
                transform = ax.transAxes)
        ax.set_ylabel('Explained Variance Ratio')
        if 'x' in label:
            ax.set_xlabel('PC dimension')
        if dim:
            ymin = dim._ev_lambda([np.min])
            ymax = dim._ev_lambda([np.max])
            ax.set_ylim([ymin, ymax+0.05])
        if 't' in label:
            plt.title('Imageset')
        
    
        
       
       
    
    
    