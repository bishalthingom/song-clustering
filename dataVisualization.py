import networkx as nx
import numpy as np
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.interpolate import interp1d

def drawGraph(distances):
    dt = [('len', float)]
    A = np.array(distances)*10
    A = A.view(dt)

    G = nx.from_numpy_matrix(A)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))

    G = nx.drawing.nx_agraph.to_agraph(G)

    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="blue", width="2.0")

    G.draw('out.png', format='png', prog='neato')

def plot2d(dists, nodes):

    adist = np.array(dists)
    amax = np.amax(adist)
    adist /= amax

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)

    coords = results.embedding_

    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        coords[:, 0], coords[:, 1], marker='o'
    )
    for label, x, y in zip(nodes, coords[:, 0], coords[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.grid()
    plt.show()

def plotLine(dists, labels):
    X = [1950, 1960, 1970, 1980, 1990, 2000]
    plt.gca().set_color_cycle(['r','r','g','g','b','b','y','y','m','m','c','c'])
    for i in range(len(dists)):
        if(i == 0 or i == (len(dists) - 1)):
        # plt.plot(X, dist)
            f1 = interp1d(X, dists[i], kind='quadratic')
            xnew = np.linspace(1950, 2000, num=41, endpoint=True)

            plt.xlabel('Decades')
            plt.ylabel('Accuracy')
            plt.plot(X, dists[i], 'o', xnew, f1(xnew), '-')
    plt.grid()
    plt.legend(['','50s', '','00s'], loc='upper center')
    plt.show()

dists = [[0.5, 0.503, 0.503, 0.546, 0.532, 0.661], [0.503, 0.5, 0.532, 0.502, 0.542, 0.567], [0.503, 0.532, 0.5, 0.512, 0.516, 0.578], [0.546, 0.502, 0.512, 0.5, 0.514, 0.551], [0.532, 0.542, 0.516, 0.514, 0.5, 0.509], [0.661, 0.567, 0.578, 0.551, 0.509, 0.5]]
nodes = []
plotLine(dists,nodes)