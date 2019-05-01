import re

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, pearsonr, kendalltau

from .experiment import fig_path
from .misc import MDS, HC

#from sklearn.manifold import MDS


class RDMVis():
    
    def __init__(self, experiment):
        super(RDMVis, self).__init__()
        self.experiment=experiment
        self.label_colors=self.experiment.label_colors
        self.labels = self.experiment.labels
        self.label_pretty = self.experiment.label_pretty
        self.label_abbr = self.experiment.label_abbr
        self.leaf_colors = self.experiment.leaf_colors
        self.leaf_default_color = self.experiment.leaf_default_color
        self.correlation_colors = self.experiment.correlation_colors
        self.num_images = self.experiment.num_images
        self.fig_path = experiment.fig_path if experiment.fig_path else fig_path
    
    def __getattr__(self, attr):
        return getattr(self.experiment, attr)
    
    def _label_pretty(self, label):
        if self.label_pretty:
            return self.label_pretty[label]
        return label
    
    def _label_abbr(self, label):
        if self.label_abbr:
            return self.label_abbr[label]
        return label
    
    def plot(self, rdm, correlation=None, save = False, show = False,
             label_rdm = 'xabyp', label_corr='y', suptitle = None):
        """Plots the RDM of all layers specified in rdm.layers and a summary
        statistics.
        """
        #self.mean()
        layers = rdm.layers
        num_columns = len(layers)
        num_rows = 4 if correlation else 3
        fig, ax = plt.subplots(num_rows, num_columns, figsize = [24, 12])
        for i, layer in enumerate(layers):
            plt.sca(ax[0, i]) #set current axis
            if i > 0:
                label_rdm = re.sub('y\w+', '',label_rdm) # rdm with abbreviations on x
                label_corr = ''    # no labels for bar plots
            self.plot_rdm(rdm, layer, label = label_rdm)
            plt.sca(ax[1, i])
            self.plot_mds(rdm, layer, title=False)
            plt.sca(ax[2, i])
            self.plot_hc(rdm, layer, title=False, cboxes = True)
            if correlation:
                plt.sca(ax[3, i])
                self.plot_corr(correlation, layer, title=False, label=label_corr)
        
        p = Pos(ax[0, -1])
        self.add_cbar([p.right+0.01, p.top-0.05, 0.007, 0.05]) # left, bottom, width, height
        handles = [mpl.patches.Patch(color=self.label_colors[img_ct],
                                     label=self._label_pretty(img_ct))
                    for img_ct in self.labels]
        labels = [self._label_pretty(label) for label in self.labels]
        p = Pos(ax[1, -1])
        self.add_legend([p.right + 0.01, p.top], ax[1, 0], fig, labels = labels,
                        handles = handles, loc = 2) #
        if correlation:
            p = Pos(ax[3, -1])
            self.add_legend([p.right + 0.01, p.top], ax[3, 0], fig, loc = 2) #
        if suptitle:
            fig.suptitle(suptitle, y = 0.9)
        if save:
            plt.savefig(self.fig_path / 
                        ('%s_%s.png'%(rdm.model._get_name(), save)),
                        dpi = 300, bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax
        
        
    def plot_rdm(self, rdm, layer, save = False, cbar = False, show = False,
                 label='xy', title=True,vmin=0.6,vmax=1.0,**kwargs):
        """Plots the RDM of the specified layer.
   
        Parameters
        ----------
        layer : str
            The layer to plot the RDM for.
        save : str
            Save the figure as rdm_save_layer.
        cbar : bool
            Plot a colorbar.
        show : bool
            Show the plot.
        label: str
            Toggle axis labeling, abbreviations and pretty labels. E.g. 'xab' shows
            abbreviated labels on the x axis, no labels on y. x[[ab][p]][y][[ab][p]]
        """

        ax = plt.gca()
        [ax.spines[key].set_visible(False) for key in ax.spines]
        ax.matshow(rdm.mean_rdm(layer), cmap='jet', vmin=vmin, vmax=vmax)
        for i in range(len(self.labels)):
            if 'y' in label:
                ylabel = (self._label_abbr(self.labels[i])
                          if 'yab' in label else self.labels[i])
                ylabel = (self._label_pretty(self.labels[i])
                          if 'yp' in label else ylabel)
                plt.text(-0.01, 1.05-(i+1)*0.1, ylabel,
                         horizontalalignment='right',
                         verticalalignment='center',
                         transform=plt.gca().transAxes,
                         color = self.label_colors[self.labels[i]])
            if 'x' in label:
                xlabel = (self._label_abbr(self.labels[i])
                          if 'xab' in label else self.labels[i])
                xlabel = (self._label_pretty(self.labels[i])
                          if 'xp' in label else xlabel)
                plt.text(-0.01+(i+1)*0.1, -0.01, xlabel,
                         horizontalalignment='right',
                         verticalalignment='top',
                          rotation = 45,
                          transform=plt.gca().transAxes,
                          color = self.label_colors[self.labels[i]])
            if not any(['x' in label, 'y' in label]):
                plt.gca().set_ylim(self.num_images, -4)
                plt.gca().set_xlim(-4, self.num_images)
                # fill along x
                plt.fill_between([i*self.num_images_per_label-0.5,
                                  (i+1)*self.num_images_per_label-0.5],
                                 [-2, -2],
                                 [-5, -5],
                                color = self.label_colors[self.labels[i]],
                                zorder = 10)
                # fill along y
                plt.fill_between([-5, -2], 
                                 [i*self.num_images_per_label-0.5,
                                  i*self.num_images_per_label-0.5],
                                 [(i+1)*self.num_images_per_label-0.5,
                                  (i+1)*self.num_images_per_label-0.5],
                                color = self.label_colors[self.labels[i]],
                                zorder = 10)
                
        ax.set_xticks(np.arange(self.num_images_per_label-0.5,
                                self.num_images,
                                self.num_images_per_label))
        ax.set_yticks(np.arange(self.num_images_per_label-0.5,
                                self.num_images,
                                self.num_images_per_label))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid('on', color = 'k')
        
        if cbar:
            fig = plt.gcf()
            cmap = mpl.cm.jet
            cbnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbax = fig.add_axes([0.91, 0.45, 0.01, 0.1])
            cbar = mpl.colorbar.ColorbarBase(cbax,
                                             cmap=cmap,
                                             norm=cbnorm,
                                     ticks = np.arange(vmin, vmax+0.1, 0.1))
            #cbar = plt.colorbar(cax, ticks = np.arange(0.6, 1.1, 0.1))
        if title:
            ax.set_title(re.sub('\d+',' \g<0>', layer).capitalize())
        if save:
            plt.savefig(self.fig_path / 'rdm_{}_{}.png'.format(save, layer), 
                        bbox_inches='tight')
        if show:
            plt.show()
        #return ax, cbar
        
    
    def plot_mds(self, rdm, layer, show = False, title = True, flip = ''):
        """Plots the MDS of the specified layer.
   
        Args:
            rdm (RDM) : The RDM object.
            layer (str) : The layer to plot the mds for.  
            show (bool) : Wether to call plt.show() at the end.
            title (bool) : Wether to set the layer string as title.
            flip (str) : Wether to flip 'x' or 'y' axis for compatibility with
                matlab.
        """
        embedding = MDS.cmdscale(rdm.mean_rdm(layer), 2)[0]
        embedding = {cat:embedding[i*self.num_images_per_label:(i+1)*self.num_images_per_label] # split into categories
                    for i, cat in enumerate(self.labels)}   
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        for cat in self.labels:
            ax.scatter(embedding[cat][:, 0],
                        embedding[cat][:, 1],
                        label = cat,
                        c = self.label_colors[cat])
        if 'x' in flip:
            ax.invert_xaxis()
        if 'y' in flip:
            ax.invert_yaxis()
        if title:
            ax.set_title(layer)# + ' Stress: {}'.format(stress))
        if show:
            plt.show()
    
    def plot_hc(self, rdm, layer, show = False, title = True, cboxes = True,
                **kwargs):
        """Plots the hierarchical clustering of the specified layer.
   
        Args:
            rdm (RDM): The RDM object.
            layer (str): The layer to plot the mds for.  
            show (bool): Wether to call plt.show() at the end.
            title (bool): Wether to set the layer string as title.
            cboxes (bool): Wether to draw colorboxes beneath the leaves.
        
        Kwargs:
            method (str): The linkage algorithm to use. See scipy's
                linkage section for full descriptions. Default is 'average'.
            optimal_ordering (bool): If True, the linkage matrix will be 
                reordered so that the distance between successive leaves is
                minimal. Defaults to True.
        """
        Y = HC.linkage(rdm.mean_rdm(layer), **kwargs)
        link_cols = HC.colorfunc(Y, self.leaf_default_color, self.leaf_colors)     
        Z = dendrogram(Y, link_color_func=lambda k: link_cols[k],
                       color_threshold = 1.37)
        idx = Z['leaves']
        ax = plt.gca()
        ax.set_xticks([])
        if cboxes:
            multiidx = [i for i in idx for j in range(10)] # x axis is scaled
            bottom = np.min(Y[:, 2])-0.1                   # by dendrogram with 
            top = np.max(Y[:, 2])+0.1                      # a factor of 10
            height = (top-bottom)*0.05
            for i, key in enumerate(multiidx):
                ax.fill_between([i, i+1],
                                [bottom, bottom],
                                [bottom+height, bottom+height],
                                color = self.leaf_colors[key], zorder = 10)
            ax.set_ylim(bottom, top)
            ax.axis('off')
        else:
            labels = [label for label in self.labels for i in range(self.num_images_per_label)]
            for i in range(self[layer].shape[0]):
                ax.text(+i/self.num_images+0.01,-0.01, labels[idx[i]],
                        rotation=90,
                        horizontalalignment='right',
                        verticalalignment='top', transform = ax.transAxes)
        if title:
            ax.set_title(layer)
        if show:
            plt.show()
        #return ax, Z
        
    def plot_corr(self, correlation, layer, show=False, title=True, label='xy'):
        """Plots the correlation of two correlated RDM objects for comparison.
        The correlation will be plottet with respect to 'layer'
        in correlation.rdm1.
   
        Args:
            correlation (Correlation): The Correlation object.
            layer (str): The layer to plot the mds for.  
            show (bool): Wether to call plt.show() at the end.
            title (bool): Wether to set the layer string as title.
            label (str): Wether to put labels on 'x' and 'y' axis.
        """
        #import pdb; pdb.set_trace()
        num_rois = len(correlation.correlation)
        num_complayers = len(correlation.rdm2.layers)
        
        mean = {}
        std = {}
        x = {}
        a = 1. 
        step = 3
        width = (step-a/2)/num_rois
        for i, roi in enumerate(correlation.correlation):
            # obtain all correlations of rdm1 layer in a list
            mean[roi] = [corr['mean'] for corr in correlation[roi][layer].values()]
            std[roi] = [corr['std'] for corr in correlation[roi][layer].values()]
            x[roi] = [a+i*width+j*step for j in range(num_complayers)]
        
        ax = plt.gca()
        error_kw = dict(ecolor='#4d5659', elinewidth=0.5, capsize=0.25)
        for roi in correlation.correlation:
            ax.bar(x[roi], height = mean[roi], width = 0.5,
                   label = self.mask_keys_pretty[roi],
                   color = correlation.colordict[roi],
                   yerr = std[roi], error_kw=error_kw) #colordict

        ax.set_ylim(correlation._corr_f(np.min), correlation._corr_f(np.max))
        
        [ax.spines[key].set_visible(False) for key in ['top', 'bottom', 'right']]
        
        ax.set_ylabel('{} Corr.'.format(correlation.method.__name__.capitalize()))
        # set x labels
        rotation = 0
        halign = 'center'
        if len(correlation.layers2)>5:
            rotation = 90
            #halign='right'
        for i, layer2 in enumerate(correlation.layers2):
            text = re.sub('\d+',' \g<0>', layer2).capitalize()
            ax.text(a+(num_rois/2-width)*width+i*step, -0.06, text,
                 horizontalalignment=halign, rotation=rotation, size='14')
        if 'x' not in label:
            ax.set_xticks([])
        if 'y' not in label:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.set_ylabel('')
        if title:
            ax.set_title(layer)
        if show:
            plt.show() 
            
    def plot_compare(self, correlation, label_rdm='', 
                     label_corr='y', save=False, show=True, 
                     row_titles = None, suptitle = None, **kwargs):
        """Plots a comparison of the RDM matrices without hierarchical clustering
        and MDS, but with correlation"""
        layers = correlation.rdm1.layers
        num_columns = len(layers)
        num_rows = 3
        fig, ax = plt.subplots(num_rows, num_columns, figsize = [24, 9])
        for i, layer in enumerate(layers):
            plt.sca(ax[0, i]) #set current axis
            if i > 0:
                label_rdm = re.sub('y\w+', '',label_rdm) # rdm with abbreviations on x
                label_corr = ''    # no labels for bar plots
            self.plot_rdm(correlation.rdm1, layer, label = label_rdm, **kwargs)
            plt.sca(ax[1, i])
            self.plot_rdm(correlation.rdm2, layer, title=False, label = label_rdm, **kwargs) # TODO: make plot of second row independent of rdm1.layers
            plt.sca(ax[2, i])
            self.plot_corr(correlation, layer, title=False, label=label_corr)
        
        p = Pos(ax[0, -1])
        self.add_cbar([p.right+0.01, p.top-0.05, 0.007, 0.05], **kwargs) # left, bottom, width, height
        
        handles = [mpl.patches.Patch(color=self.label_colors[img_ct],
                                     label=self._label_pretty(img_ct))
                    for img_ct in self.labels]
        labels = [self._label_pretty(label) for label in self.labels]
        #p = Pos(ax[1, -1])
        self.add_legend([p.right + 0.01, p.top-0.08], ax[0, 0], fig,
                        labels = labels, handles = handles, loc = 2) 
        p = Pos(ax[2, -1])
        self.add_legend([p.right + 0.01, p.top], ax[2, 0], fig, loc = 2)
        if row_titles:
            ax[0, 0].set_ylabel(row_titles[0])
            ax[1, 0].set_ylabel(row_titles[1])
        if suptitle:
            fig.suptitle(*suptitle)
        if save:
            plt.savefig(self.fig_path / 
                        ('%s_rdacomparison_%s.png'%(correlation.rdm1.model._get_name(), save)),
                        dpi = 300, bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax
    
    def plot_layer(self, rdm, correlation, layer, save = False):
        '''Plots the summary statistic for layer.
        '''

        fig, ax = plt.subplots(2, 2, figsize = [12, 12])
        plt.sca(ax[0, 0]) #set current axis
        self.plot_rdm(rdm, layer, label = 'xpyp', title = False)
        plt.sca(ax[0, 1])
        self.plot_mds(rdm, layer, title=False)
        plt.sca(ax[1, 0])
        self.plot_hc(rdm, layer, title=False, cboxes = True)
        plt.sca(ax[1, 1])
        self.plot_corr(correlation, layer, label = 'y', title = False)
        plt.subplots_adjust(hspace=0.2)
        pos = Pos(ax[0, 0])
        fig.suptitle(re.sub('\d',' \g<0>', layer).capitalize(),
                     y = pos.top+0.03)
        self.add_cbar([pos.right+0.01, pos.top-0.05, 0.007, 0.05]) # left, bottom, width, height
        handles = [mpl.patches.Patch(color=self.label_colors[img_ct],
                                     label=self._label_pretty(img_ct))
                    for img_ct in self.labels]
        labels = [self._label_pretty(label) for label in self.labels]
        pos = Pos(ax[0, 1])
        self.add_legend([pos.right+0.01, pos.top], ax[0, 1], fig, 
                        handles = handles, labels = labels, loc = 2) #
        pos = Pos(ax[1, 1])
        self.add_legend([pos.right+0.01, pos.top], ax[1, 1], fig, loc = 2)
        if save:
            plt.savefig(self.fig_path / ('%s_%s_%s.png'%(correlation.rdm1.model._get_name(), save, layer)),
                        dpi = 300, bbox_inches='tight')
        plt.show()
        
    def add_cbar(self, pos = [0.91, 0.45, 0.01, 0.1], cmap = mpl.cm.jet,
                     vmin=0.6, vmax=1.0, **kwargs):
        fig = plt.gcf()
        cmap = cmap
        cbnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbax = fig.add_axes(pos)
        steps = 0.1 if (vmax-vmin) <= 0.5 else 0.2
        ticks = np.arange(vmin, vmax+steps, steps)
        mpl.colorbar.ColorbarBase(cbax,
                                     cmap=cmap,
                                     norm=cbnorm,
                                     ticks = ticks)
        
    def add_legend(self, position, ax, fig, handles = None, labels = None, **kwargs):
        
        h, l = ax.get_legend_handles_labels()
        if not handles:
            handles = h
        if not labels:
            labels = l
        fig.legend(handles, labels, bbox_to_anchor = position,
            bbox_transform = fig.transFigure, **kwargs)

    
class Pos():
    
    def __init__(self, ax):
        self.ax = ax
        self.p0, self.p1 = ax.get_position().get_points()
        self.left = self.p0[0]
        self.bottom = self.p0[1]
        self.right = self.p1[0]
        self.top = self.p1[1]
        
