import numpy as np
import matplotlib.pyplot as plt

from utilities import *




def heterozygosity_map(chromosome, fname = None):
    ref, alt = read_data('./Data/glioblastoma_BT_S2/ref.csv', './Data/glioblastoma_BT_S2/alt.csv')
    
    ref_proportion = (ref + 1) / (ref + alt + 2) # add a dummy count to both ref and alt to avoid division by 0
    alpha = 2 * np.arctan(ref + alt) / np.pi # hide loci without enough counts
    
    plt.figure(figsize=(24,16))
    plt.imshow(ref_proportion, cmap = 'viridis', vmin = 0., vmax = 1., alpha = alpha) 
    # "viridis": yellow for 1, purple for 0, green/blue for 0.5 (https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html)
    plt.title(chromosome, fontsize = 17)
    plt.xlabel('locus index', fontsize = 17)
    plt.ylabel('cell index', fontsize = 17)
    plt.tight_layout()
    if fname is None:
        fname = 'map_' + chromosome + '.png'
    plt.savefig('./figures/' + fname)

    
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

        
def mix_columns(arr1, arr2):
    result = np.empty((arr1.shape[0], 2*arr1.shape[1]))
    for j in range(arr1.shape[1]):
        result[:,2*j] = arr1[:,j]
        result[:,2*j+1] = arr2[:,j]
    return result



if __name__ == '__main__': 
    # Plot results of space swap test
    heterozygosity_map('chr1')

    '''
    data = np.empty((3,3), dtype = np.ndarray)
    data[0,0] = np.load('./test_results/space_swap_L_100c_50m_50f.npy')
    data[0,1] = np.load('./test_results/space_swap_L_100c_100m_100f.npy')
    data[0,2] = np.load('./test_results/space_swap_L_100c_200m_200f.npy')
    data[1,0] = np.load('./test_results/space_swap_D_100c_50m_50f.npy')
    data[1,1] = np.load('./test_results/space_swap_D_100c_100m_100f.npy')
    data[1,2] = np.load('./test_results/space_swap_D_100c_200m_200f.npy')
    data[2,0] = np.load('./test_results/space_swap_T_100c_50m_50f.npy')
    data[2,1] = np.load('./test_results/space_swap_T_100c_100m_100f.npy')
    data[2,2] = np.load('./test_results/space_swap_T_100c_200m_200f.npy')

    fig, axes = plt.subplots(3, 3, sharex = 'all', sharey = 'row', figsize = (12,10), dpi = 300)
    titles = [
        '100 cells\n 50 true mutations\n 50 fake mutations',
        '100 cells\n 100 true mutations\n 100 fake mutations',
        '100 cells\n 200 true mutations\n 200 fake mutations'
    ]
    xlabels = ['C', 'M', 'CM', 'MC']
    ylabels = [
        'mean loglikelihood of inferred tree \n compared to real tree', 
        'MSE of distance matrix', 
        'runtime (s)'
    ]
    colors = ['skyblue', 'gold', 'green', 'limegreen']
    for i in range(3): 
        axes[i,0].set_ylabel(ylabels[i], fontsize = 11)
        for j in range(3): 
            make_boxplot(axes[i,j], data[i,j], colors = colors)
            axes[0,j].set_title(titles[j], fontsize = 11)
            axes[2,j].set_yscale('log') 
            axes[2,j].set_xticks(np.arange(1,5), labels = xlabels)
    
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig('./figures/space_swap.png')
    '''

    # plot results of mutationd detection test
    fig, axes = plt.subplots(4, 3, figsize = (14,16), dpi = 300)
    
    plot_mut_detection_results(25, 'def', axes[0,:])
    plot_mut_detection_results(400, 'def', axes[1,:])
    plot_mut_detection_results(25, 'oth', axes[2,:])
    plot_mut_detection_results(400, 'oth', axes[3,:])
    
    row_names = [   
        '25 cells\n Poisson coverage', 
        '400 cells\n Poisson coverage', 
        '25 cells\n data coverage', 
        '400 cells\n data coverage'
    ]
    for ax, rn in zip(axes[:,0], row_names): 
        ax.annotate(rn, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center', color = 'red')
    
    fig.tight_layout()
    fig.savefig('./figures/mut_detection.png')

    '''
    data = np.empty((3,2), dtype = np.ndarray)
    path = ['./test_results/thresholds_50c_200m_200f_rev/', './test_results/thresholds_50c_200m_200f_irr/']
    data[0,0] = np.load(path[0] + 'MSE.npy')
    data[0,1] = np.load(path[1] + 'MSE.npy')
    data[1,0] = np.load(path[0] + 'n_selected.npy')
    data[1,1] = np.load(path[1] + 'n_selected.npy')
    data[2,0] = np.load(path[0] + 'runtime.npy')
    data[2,1] = np.load(path[1] + 'runtime.npy')
    
    dist = mix_columns(data[0,0], data[0,1])
    n_selected = mix_columns(data[1,0], data[1,1])
    runtime = mix_columns(data[2,0], data[2,1])

    fig, axes = plt.subplots(3, 1, figsize = (9,12), dpi = 300)
    thresholds = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    ylabels = [
        'MSE of distance matrix', 
        'number of selected loci', 
        'runtime (s)'
    ]
    colors = ['pink', 'lightblue'] * 7
    positions = np.array([1/8, -1/8] * 7) + np.arange(1, 14+1)

    make_boxplot(axes[0], dist, colors = colors, positions = positions)
    make_boxplot(axes[1], n_selected, colors = colors, positions = positions)
    make_boxplot(axes[2], runtime, colors = colors, positions = positions)

    for i in range(3):
        axes[i].set_ylabel(ylabels[i], fontsize = 11)
        axes[i].set_xticks(np.arange(1,8) * 2 - 0.5, labels = thresholds)
    axes[0].set_ylim(5, 55)
    axes[2].set_xlabel('threshold')
    axes[2].set_yscale('log')
    
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig('./figures/thresholds.png')
    '''

    path = './test_results/reversibility_50c_100m/'
    likelihoods_rev = np.load(path + 'likelihoods_rev.npy')
    likelihoods_irr = np.load(path + 'likelihoods_irr.npy')
    dist_rev = np.load(path + 'dist_rev.npy')
    dist_irr = np.load(path + 'dist_irr.npy')
    
    likelihoods = mix_columns(likelihoods_rev, likelihoods_irr)
    dist = mix_columns(dist_rev, dist_irr)

    fig, axes = plt.subplots(2, 1, figsize = (9,12), dpi = 300)
    wrong_prop = [0.0, 0.25, 0.5, 0.75, 1.0]
    ylabels = [
        'mean loglikelihood\n compared to real tree', 
        'MSE of distance matrix'
    ]
    colors = ['pink', 'lightblue'] * 5
    positions = np.array([1/8, -1/8] * 5) + np.arange(1, 10+1)

    make_boxplot(axes[0], likelihoods, colors = colors, positions = positions)
    make_boxplot(axes[1], dist, colors = colors, positions = positions)

    for i in range(2):
        axes[i].set_ylabel(ylabels[i], fontsize = 11)
        axes[i].set_xticks(np.arange(1,6) * 2 - 0.5, labels = wrong_prop)
    axes[1].set_xlabel('proportion of wrong directions')
    
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig('./figures/reversibility.png')


    '''
    data = np.load('./test_results/random_dist.npy')
    fig, ax = plt.subplots(figsize = (3,4), dpi = 300)
    make_boxplot(ax, data)
    #ax.set_title('distance between ramdom trees')
    ax.set_xlabel('tree size (number of cells)')
    ax.set_ylabel('MSE of distance matrix')
    ax.set_xticklabels([25, 50, 100, 200])
    fig.tight_layout()
    fig.savefig('./figures/random_dist.png')
    '''