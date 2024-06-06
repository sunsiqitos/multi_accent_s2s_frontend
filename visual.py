import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_alignment(alignment, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(
        alignment.T, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    # plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    return fig


def plot_pca(embeddings, idx2token, n_components=2):
    labels = [idx2token[idx] for idx in range(len(idx2token))]
    embeddings = np.array(embeddings)
    assert len(idx2token) == embeddings.shape[0], "Dictionary is not of the same length as the Embedding size"

    if n_components == 2 and len(labels) >= 2:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(16, 10))
        for idx, (x, y) in enumerate(pca_results):
            plt.scatter(x, y, alpha=0.8, marker='x', color='b')
            plt.annotate(labels[idx], (x, y + 0.2))
        plt.title('Embedding 2D PCA result, variation: {}'.format(pca.explained_variance_ratio_))
        # plt.legend()
        plt.tight_layout()
        return fig
    elif n_components == 3 and len(labels) >= 3:
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        for idx, (x, y, z) in enumerate(pca_results):
            ax.scatter(x, y, z, alpha=0.8, marker='x', color='b')
            ax.text(x, y, z, '%s' % (labels[idx]), size=20, zorder=1, color='k')
        ax.set_title('Embedding 3D PCA result, variation: {}'.format(pca.explained_variance_ratio_))
        # ax.legend()
        plt.tight_layout()
        return fig


def plot_tsne(embeddings, idx2token, n_components=2):
    labels = [idx2token[idx] for idx in range(len(idx2token))]
    embeddings = np.array(embeddings)
    assert len(idx2token) == embeddings.shape[0], "Dictionary is not of the same length as the Embedding size"

    if n_components == 2 and len(labels) >= 2:
        tsne = TSNE(n_components=2, verbose=0, perplexity=3, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(16, 10))
        for idx, (x, y) in enumerate(tsne_results):
            plt.scatter(x, y, alpha=0.8, marker='x', color='b')
            plt.annotate(labels[idx], (x, y + 0.3))
        plt.title('Embedding 2D t-SNE result')
        # plt.legend()
        plt.tight_layout()
        return fig
