import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

def featuresPlot(data, rowVar=False, featNames=None, histogramBins=10, classLabels=None):
    """
    only works for 2-D data

    rowVar: boolean
        if rowVar is true data should have dimensions #feats x #samples
        otherwise  #samples x  #features (default)



    """
    if rowVar:
        data = data.T

    d = data.shape[1]

    if type(histogramBins) is int:
        histogramBins = [histogramBins] * d

    if featNames is not None:
        assert len(featNames) == d

    else:
        featNames = ['feature ' + str(i+1) for i in range(d)]

    if classLabels is not None:
        assert len(classLabels) == data.shape[0]

    fig = plt.figure()

    for i in range(d):
        for j in range(i, d):

            ax = fig.add_subplot(d,d, 1+j*d+i)

            if i == j:
                ax.hist(data[:,j], bins=histogramBins[i], edgecolor='k', density=True)
                ax.set_xlabel(featNames[i])
                ax.set_ylabel('density')

            else:
                if classLabels is None:
                    ax.scatter(data[:,i], data[:,j], edgecolor='k')
                else:
                    scatterColorByGroup(data[:, [i,j]], classLabels, ax=ax)
                ax.set_xlabel(featNames[i])
                ax.set_ylabel(featNames[j])



    plt.tight_layout()
    return fig


def scatterColorByGroup(x, y, colors=None, labels=None, ax=None):
    """
    :param x:
    :param y:
    :param colors:
    :param labels:
    :return:
    """

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 2

    nGroups = len(set(y))

    assert nGroups < 11

    if colors is None:
        colors = ['C' + str(i) for i in range(nGroups)]

    else:
        assert len(colors) == nGroups

    if labels is None:
        labels = ['Class ' + str(i+1) for i in range(nGroups)]
    else:
        assert len(labels) == nGroups

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, g in enumerate(set(y)):
            ax.scatter(x[y==g,0], x[y==g,1], color=colors[i], label=labels[i], edgecolor='k')


        plt.legend()


    return ax


