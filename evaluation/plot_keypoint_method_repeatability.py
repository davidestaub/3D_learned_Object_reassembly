import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_dists(dists, df, method, ax):
    data = dists[df['method'] == method].reshape(-1)
    ax.hist(data, bins=np.linspace(0, np.nanmax(dists), 100), weights=np.ones_like(data) / len(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim((0, np.nanmax(dists)))
    # ax.set_ylim((0, 0.07))
    ax.set_xlabel('Distance')
    ax.set_ylabel('# points')
    ax.set_title(f'Method: {method}, n={len(data)}')


def plot_cumulative_dists(dists, df, method, ax, title=''):
    data = dists[df['method'] == method].reshape(-1)
    n, bins, patch = ax.hist(data,
            bins=np.linspace(0, np.nanmax(dists), 500),
            density=True,
            cumulative=True,
            histtype='step',
            label=method)
    bin_size = bins[1] - bins[0]
    idx_1 = np.argmax(bins > 1.0)
    auc = (bin_size * n[:idx_1]).sum()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim((0, 1.0))
    # ax.set_ylim((0, 0.07))
    ax.set_xlabel('Distance')
    ax.set_ylabel('Cumulative % of points')
    ax.set_title(title)
    return patch[0], f'{method}, AUC={auc:.2}'


def plot_surface():
    csv_path = 'keypoint_surface_repeatability.csv'
    df = pd.read_csv(csv_path)
    dists = df.loc[:, 'k_0':].to_numpy()
    df = df.loc[:, :'fragment_2']
    methods = df['method'].unique()

    fig = plt.figure(figsize=(6, 12))
    axes = fig.subplots(len(methods), 1, sharex=True)
    [plot_dists(dists, df, method, ax) for method, ax in zip(methods, axes)]
    plt.savefig('keypoint_surface_repeatability.svg')
    fig.show()

    fig = plt.figure(figsize=(4, 4))
    patches, labels = zip(*([plot_cumulative_dists(dists, df, method, fig.gca(), 'Surface keypoint adjacency') for method in methods]))
    fig.gca().legend(handles=patches, labels=labels, loc='lower right')
    plt.grid(visible=True)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.93)
    plt.savefig('keypoint_surface_repeatability_cumulative.svg', bbox_inches='tight')
    fig.show()

    r = 0.05
    num_pts = dists.shape[1]  # dists.shape = [2 * NUM_PTS * NUM_METHODS, NUM_FRAGMENTS]
    df['rel_repeatability'] = (dists < r).sum(axis=1) / num_pts
    print(df.groupby('method')['rel_repeatability'])
    df.boxplot(column='rel_repeatability', by='method')
    plt.show()
    print(df.describe())
    print(df)


def plot():
    csv_path = 'keypoint_repeatability.csv'
    df = pd.read_csv(csv_path)
    dists = df.loc[:, 'k_0':].to_numpy()
    df = df.loc[:, :'method']
    methods = df['method'].unique()

    fig = plt.figure(figsize=(6, 12))
    axes = fig.subplots(len(methods), 1, sharex=True)
    [plot_dists(dists, df, method, ax) for method, ax in zip(methods, axes)]
    plt.savefig('keypoint_repeatability.svg')
    fig.show()

    fig = plt.figure(figsize=(4, 4))
    patches, labels = zip(*([plot_cumulative_dists(dists, df, method, fig.gca(), 'Keypoint repeatability') for method in methods]))
    fig.gca().legend(handles=patches, labels=labels, loc='lower right')
    plt.grid(visible=True)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.93)
    plt.savefig('keypoint_repeatability_cumulative.svg', bbox_inches='tight')
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
    plot_surface()
    plot()
