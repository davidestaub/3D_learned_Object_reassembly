import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot():
    csv_path = 'keypoint_repeatability.csv'
    df = pd.read_csv(csv_path)
    dists = df.loc[:, 'k_0':].to_numpy()

    def plot_dists(method, ax):
        data = dists[df['method'] == method].reshape(-1)
        ax.hist(data, bins=np.linspace(0, np.nanmax(dists), 100), weights=np.ones(len(data)) / len(data))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim((0, np.nanmax(dists)))
        ax.set_ylim((0, 0.07))
        ax.set_xlabel('Distance')
        ax.set_ylabel('# points')
        ax.set_title(f'Method: {method}, n={len(data)}')

    methods = df['method'].unique()

    fig = plt.figure(figsize=(6, 12))
    axes = fig.subplots(len(methods), 1, sharex=True)
    [plot_dists(method, ax) for method, ax in zip(methods, axes)]
    plt.savefig('keypoint_repeatability.png')
    fig.show()

    r = 0.05
    num_pts = dists.shape[1] # dists.shape = [2 * NUM_PTS * NUM_METHODS, NUM_FRAGMENTS]
    df['rel_repeatability'] = (dists < r).sum(axis=1) / num_pts
    print(df.describe())
    print(df)


if __name__ == '__main__':
    plot()