import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

RMAX = 0.1


def plot_dists(dists, df, method, ax):
    data = dists[df['method'] == method].reshape(-1)
    ax.hist(data, bins=np.linspace(0, np.nanmax(dists), 1000), weights=np.ones_like(data) / len(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim((0, np.nanmax(dists)))
    # ax.set_ylim((0, 0.07))
    ax.set_xlabel('Distance')
    ax.set_ylabel('# points')
    ax.set_title(f'Method: {method}, n={len(data)}')


def plot_cumulative_dists(dists, df, method, ax):
    data = dists[df['method'] == method].reshape(-1)
    n, bins, patch = ax.hist(data[~np.isnan(data)],
                             bins=np.linspace(0, np.nanmax(data), 1000),
                             density=True,
                             cumulative=True,
                             histtype='step',
                             label=method)
    bin_size = bins[1] - bins[0]
    idx_xlim = np.argmax(bins > RMAX)
    auc = (bin_size * n[:idx_xlim]).sum()
    return patch[0], f'{method}, AUC={auc:.2}', n[idx_xlim]


def plot(name, last_info_column, title):
    csv_path = f'{name}.csv'
    df = pd.read_csv(csv_path)
    dists = df.loc[:, 'k_0':].to_numpy()
    df = df.loc[:, :last_info_column]
    methods = df['method'].unique()

    fig = plt.figure(figsize=(6, 12))
    axes = fig.subplots(len(methods), 1, sharex=True)
    [plot_dists(dists, df, method, ax) for method, ax in zip(methods, axes)]
    plt.savefig(f'{name}.pdf')
    fig.show()

    fig = plt.figure(figsize=(3.5, 3.5))
    patches, labels, ymaxlim = zip(*([plot_cumulative_dists(dists, df, method, fig.gca()) for method in methods]))
    ax = fig.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim((0, RMAX))
    ax.set_ylim((0, 1.05 * max(ymaxlim)))
    ax.set_xlabel('Distance')
    ax.set_ylabel('Cumulative % of points')
    ax.set_title(title)
    ax.legend(handles=patches, labels=labels, loc='upper left')
    plt.grid(visible=True)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.93)
    plt.savefig(f'{name}_cumulative.pdf', bbox_inches='tight')
    fig.show()

    r = 0.05
    num_pts = dists.shape[1]  # dists.shape = [2 * NUM_PTS * NUM_METHODS, NUM_FRAGMENTS]
    df['rel_repeatability'] = (dists < r).sum(axis=1) / num_pts
    print(df.groupby('method')['rel_repeatability'])
    df.boxplot(column='rel_repeatability', by='method')
    plt.show()
    print(df.describe())
    print(df)


if __name__ == '__main__':
    plot('keypoint_surface_repeatability', last_info_column='fragment_2', title='Repeatability on fracture surface')
    plot('keypoint_repeatability', last_info_column='method', title='Keypoint repeatability')
