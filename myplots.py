import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.stats import gaussian_kde

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


def scatterColorByGroup(x, y=None, colors=None, labels=None, ax=None, alpha=1.0):
    """
    :param x:
    :param y:
    :param colors:
    :param labels:
    :return:
    """

    if y is None:
        y = np.zeros(x.shape[0], int)
        labels = ['']

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 2 or x.shape[1] == 3

    nGroups = len(set(y))

    assert nGroups < 50

    if colors is None:
        nn = min(nGroups, 10)
        colors = ['C' + str(i) for i in range(nn)]

    else:
        assert len(colors) == nGroups

    if nGroups == len(colors):
        markers = ['o']
    else:
        markers = ['o', 's', 'D', 'X', '+']


    if labels is None:
        labels = ['Class ' + str(i+1) for i in range(nGroups)]
    else:
        assert len(labels) == nGroups

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i, g in enumerate(sorted(set(y))):

        ax.scatter(x[y==g,0], x[y==g,1], color=colors[i % len(colors)], marker=markers[i // len(colors)], label=labels[i], s=5, alpha=alpha)


        plt.legend()


    return ax


def dimRedPlot(X, method, y=None, ax=None, threeD=False, classNames=None, showPoints=True, showDensity=True, tsneSeed=17021991, colormap='Oranges'):
    """
    :param X:
    :param method:
    :param y:
    :param ax:
    :param threeD:
    :param classNames:
    :return: reducer
    :return: ax
    """

    assert showPoints or showDensity

    if ax is None:
        fig = plt.figure()
        if threeD:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    if threeD:
        comp = 3
        assert not showDensity
    else:
        comp = 2

    np.random.seed(tsneSeed)
    if method.lower() == 'pca':
        reducer = PCA(n_components=comp)
    elif method.lower() == 'tsne':
        if X.shape[1] > 150:
            pca = PCA(n_components=150)
            X = pca.fit_transform(X)
        reducer = TSNE(n_components=comp, verbose=1)
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=comp)
    else:
        raise ValueError('Invalid or not supported dim red method. Use pca/tsne/umap')

    pcs = reducer.fit_transform(X)

    alpha = 1
    if showDensity:
        print('density estimation starts')
        _, ax = density2D(pcs, ax=ax, colormap=colormap)
        alpha = 0.3

    if showPoints:
        ax = scatterColorByGroup(pcs, y, ax=ax, labels=classNames, alpha=alpha)

    return reducer, ax


def density2D(data, ax=None, Npoints=500, colormap='Blues', showContours=False):
    """

    Parameters
    ----------
    data
    ax
    Npoints
    colormap
    showContours

    Returns
    -------

    """

    assert data.shape[1] == 2

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    xmin, ymin = np.min(data, axis=0)
    xmax, ymax = np.max(data, axis=0)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:complex(0,Npoints), ymin:ymax:complex(0,Npoints)]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    kernel = gaussian_kde(data.T)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Contourf plot
    ax.contourf(xx, yy, f, cmap=colormap, levels=15)
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(np.rot90(f), cmap=colormap, extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    if showContours:
        ax.contour(xx, yy, f, colors='k')


    return f, ax
