import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression

from utilities import *




def heterozygosity_map(chromosome, fname = None):
    ref, alt = read_data('./Data/glioblastoma_BT_S2/ref.csv', './Data/glioblastoma_BT_S2/alt.csv', chromosome)
    
    ref_proportion = (ref + 1) / (ref + alt + 2) # add a dummy count to both ref and alt to avoid division by 0
    alpha = 2 * np.arctan(ref + alt) / np.pi # hide loci without enough counts
    
    plt.figure(figsize=(12,8), dpi = 300)
    plt.imshow(ref_proportion, cmap = 'viridis', vmin = 0., vmax = 1., alpha = alpha) 
    # "viridis": yellow for 1, purple for 0, green/blue for 0.5 (https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html)
    plt.title(chromosome + ' heterozygosity', fontsize = 17)
    plt.xlabel('locus', fontsize = 17)
    plt.ylabel('cell', fontsize = 17)
    plt.tight_layout()
    if fname is None:
        fname = chromosome + '_heterozygosity.png'
    plt.savefig('./figures/' + fname)
    plt.close()

    
def make_boxplot(ax, data, colors = None, positions = None):
    ax.set_facecolor('lightgray')
    bplot = ax.boxplot(data, patch_artist = True, positions = positions)
    if colors is None:
        colors = 'lightblue'
    if type(colors) == str:
        colors = [colors] * data.shape[0]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(color = 'white')
    return bplot


def make_roc_curve(axes, sorted_posteriors, tpr, fpr, tpr_alt, fpr_alt):
    n_tests = tpr.shape[0]
    n_loci = tpr.shape[1]

    color = 'lightblue'
    alpha = 0.2 #min(5 / n_tests, 1.)

    thresholds = np.concatenate((np.zeros((n_tests,1)), sorted_posteriors), axis = 1)
    for i in range(tpr.shape[0]): 
        axes[0].plot(thresholds[i,:], tpr[i,:], c = color, alpha = alpha)
        axes[1].plot(thresholds[i,:], fpr[i,:], c = color, alpha = alpha)
        axes[2].plot(fpr[i,:], tpr[i,:], c = color, alpha = alpha, zorder = 2)

    for ax in axes: 
        ax.grid(True, c = 'lightgray')

    axes[0].plot(np.mean(thresholds, axis = 0), np.mean(tpr, axis = 0), c = 'darkblue', lw = 2.)
    axes[0].set_xlabel('threshold')
    axes[0].set_ylabel('TPR (sensitivity)')

    axes[1].plot(np.mean(thresholds, axis = 0), np.mean(fpr, axis = 0), c = 'darkblue', lw = 2.)
    axes[1].set_xlabel('threshold')
    axes[1].set_ylabel('FPR (fall-out)')

    axes[2].scatter(fpr_alt, tpr_alt, s = 20, c = 'none', edgecolors = 'orange', label = 'highest posterior', alpha = 0.5, zorder = 3)
    axes[2].scatter(fpr[:,n_loci//2], tpr[:,n_loci//2], s = 20, c = 'none', edgecolors = 'mediumseagreen', label = 'best 50%', alpha = 0.5, zorder = 3)
    axes[2].legend()

    axes[2].plot(np.mean(fpr, axis = 0), np.mean(tpr, axis = 0), c = 'darkblue', lw = 2., zorder = 3)
    axes[2].plot([0, 1], [0, 1], c = "r", linestyle = "--", zorder = 1)
    axes[2].set_xlabel('FPR (fall-out)')
    axes[2].set_ylabel('TPR (sensitivity)')


def plot_mut_detection_results(n_cells, sampler_name, axes): 
        sorted_posteriors = np.load('./test_results/mut_detection_sortedP_%ic_100m_100f_%s.npy' % (n_cells, sampler_name))
        tpr = np.load('./test_results/mut_detection_TPR_%ic_100m_100f_%s.npy' % (n_cells, sampler_name))
        fpr = np.load('./test_results/mut_detection_FPR_%ic_100m_100f_%s.npy' % (n_cells, sampler_name))
        tpr_alt = np.load('./test_results/mut_detection_altTPR_%ic_100m_100f_%s.npy' % (n_cells, sampler_name))
        fpr_alt = np.load('./test_results/mut_detection_altFPR_%ic_100m_100f_%s.npy' % (n_cells, sampler_name))

        make_roc_curve(axes, sorted_posteriors, tpr, fpr, tpr_alt, fpr_alt)

        
def coverage_hist(ax, coverages):
    ax.set_facecolor('lightgray')
    bins = np.arange(coverages.max() + 1)
    ax.hist(coverages, bins = bins)
    ax.plot(bins, coverages.size * poisson.pmf(bins, mu = np.mean(coverages)), label = 'poisson with same mean')
    ax.legend()
    ax.grid(True, color = 'white', alpha = 0.6)
    ax.set_xlabel('coverage')
    ax.set_ylabel('count')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xlim(10**(-0.5), coverages.max() + 1)
    ax.set_ylim(10**(-0.5), 1e8)

    
def dist_vs_likelihood(ax, x, y, dot_color):
    ax.set_facecolor('lightgray')
    ax.scatter(x, y, s = 20, c = dot_color, alpha = 0.5)
    ax.axvline(0, c = 'black', ls = 'dotted')
    reg_x = np.expand_dims(x.flatten(), axis = 1)
    reg_y = y.flatten()
    reg = LinearRegression().fit(reg_x, reg_y)
    ax.axline([0, reg.intercept_], slope = reg.coef_[0], c = 'red', ls = 'dashed')
    ax.grid(True, c = 'white')
        

def mix_columns(arr1, arr2):
    result = np.empty((arr1.shape[0], 2*arr1.shape[1]))
    for j in range(arr1.shape[1]):
        result[:,2*j] = arr1[:,j]
        result[:,2*j+1] = arr2[:,j]
    return result