import numpy as np
import matplotlib.pyplot as plt


def make_boxplot(ax, data): 
    ax.set_facecolor('lightgray')
    bplot = ax.boxplot(data, patch_artist = True)
    for patch in bplot['boxes']: 
        patch.set_facecolor('lightblue')
    ax.yaxis.grid(color = 'white')
    return bplot


def make_roc_curve(axes, sorted_posteriors, tpr, fpr, tpr_alt, fpr_alt): 
    n_tests = tpr.shape[0]
    n_loci = tpr.shape[1]

    color = 'lightblue'
    alpha = 0.2 #min(5 / n_tests, 1.)

    thresholds = np.concatenate((np.zeros((n_tests,1)), sorted_posteriors, np.ones((n_tests,1))), axis = 1)
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

    axes[2].scatter(fpr_alt, tpr_alt, c = 'none', edgecolors = 'orange', zorder = 3)
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




if __name__ == '__main__': 
    import pandas as pd

    # Plot results of space swap test
    data = np.empty((3,3), dtype = pd.DataFrame)
    data[0,1] = np.load('./test_results/space_swap_L_100c_100m_100f.npy')
    data[0,0] = np.load('./test_results/space_swap_L_100c_50m_50f.npy')
    data[0,2] = np.load('./test_results/space_swap_L_100c_200m_200f.npy')
    data[1,0] = np.load('./test_results/space_swap_D_100c_50m_50f.npy')
    data[1,1] = np.load('./test_results/space_swap_D_100c_100m_100f.npy')
    data[1,2] = np.load('./test_results/space_swap_D_100c_200m_200f.npy')
    data[2,0] = np.load('./test_results/space_swap_T_100c_50m_50f.npy')
    data[2,1] = np.load('./test_results/space_swap_T_100c_100m_100f.npy')
    data[2,2] = np.load('./test_results/space_swap_T_100c_200m_200f.npy')

    fig, axes = plt.subplots(3, 3, sharex = 'all', sharey = 'row', figsize = (12,10), dpi = 300)
    titles = ['100 cells\n 50 true mutations\n 50 fake mutations',
              '100 cells\n 100 true mutations\n 100 fake mutations',
              '100 cells\n 200 true mutations\n 200 fake mutations'
             ]
    xlabels = ['C', 'CM', 'M', 'MC']
    ylabels = ['mean loglikelihood of inferred tree \n compared to real tree', 
               'MSE of distance matrix', 
               'runtime (s)'
              ]
    for i in range(3): 
        axes[i,0].set_ylabel(ylabels[i], fontsize = 11)
        for j in range(3): 
            bplot = make_boxplot(axes[i,j], data[i,j])
            axes[0,j].set_title(titles[j], fontsize = 11)
            axes[2,j].set_yscale('log') 
            axes[2,j].set_xticks(np.arange(1,5), labels = xlabels)
    
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig('./figures/space_swap.png')




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