import matplotlib.pyplot as plt
import numpy as np
import os



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


def make_test_plots(outdir):
    plotdir = os.path.join(outdir, 'plots')
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    runtime = np.loadtxt(os.path.join(outdir, 'runtime.txt'))
    dist = np.loadtxt(os.path.join(outdir, 'distance.txt'))
    llh_diff = np.loadtxt(os.path.join(outdir, 'llh_diff.txt'))
    
    fig, axes = plt.subplots(3, 1, figsize = (6,9), sharex=True, dpi=300)

    make_boxplot(axes[0], runtime)
    make_boxplot(axes[1], dist)
    make_boxplot(axes[2], llh_diff)

    axes[0].set_ylabel('runtime')
    axes[1].set_ylabel('distance to real tree')
    axes[2].set_ylabel('log-likelihood')
    fig.align_ylabels(axes)

    if os.path.exists(os.path.join(outdir, 'setting_names.txt')):
        setting_names = np.loadtxt(os.path.join(outdir, 'setting_names.txt'), delimiter='\n', dtype=str)
        axes[-1].set_xticks(np.arange(1, 4), setting_names)

    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'result.pdf'))
    
    