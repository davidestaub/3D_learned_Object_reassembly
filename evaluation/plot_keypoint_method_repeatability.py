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
    ax.hist(data,
            bins=np.linspace(0, np.nanmax(dists), 500),
            density=True,
            cumulative=True,
            histtype='step',
            label=method)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim((0, 1.0))
    # ax.set_ylim((0, 0.07))
    ax.set_xlabel('Distance')
    ax.set_ylabel('# points')
    ax.set_title(title)


def plot_surface():
    csv_path = 'keypoint_surface_repeatability.csv'
    df = pd.read_csv(csv_path)
    dists = df.loc[:, 'k_0':].to_numpy()
    df = df.loc[:, :'fragment_2']
    methods = df['method'].unique()

    fig = plt.figure(figsize=(6, 12))
    axes = fig.subplots(len(methods), 1, sharex=True)
    [plot_dists(dists, df, method, ax) for method, ax in zip(methods, axes)]
    plt.savefig('keypoint_surface_repeatability.png')
    fig.show()

    fig = plt.figure(figsize=(6, 4))
    [plot_cumulative_dists(dists, df, method, fig.gca()) for method in methods]
    fig.gca().legend(loc='lower right')
    plt.grid(visible=True)
    plt.savefig('keypoint_surface_repeatability_cumulative.png')
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
    plt.savefig('keypoint_repeatability.png')
    fig.show()

    fig = plt.figure(figsize=(6, 4))
    [plot_cumulative_dists(dists, df, method, fig.gca()) for method in methods]
    fig.gca().legend(loc='lower right')
    plt.grid(visible=True)
    plt.savefig('keypoint_repeatability_cumulative.png')
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
