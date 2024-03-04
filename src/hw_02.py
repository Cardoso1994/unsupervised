import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix)



data = np.loadtxt('../data/zoo/zoo.data', delimiter=',', dtype='object')
data = data[:, :-1]

_animals = list(data[:, 0])
animals = {_name_: i for i, _name_ in enumerate(_animals)}
vecs = data[:, 1:].astype(int)

confs = [
    {'n_clusters': 7, 'linkage': 'ward'},
    {'n_clusters': 7, 'linkage': 'complete'},
    {'n_clusters': 7, 'linkage': 'average'},
    {'n_clusters': 7, 'linkage': 'single'},
]

cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean',
                                  compute_full_tree='auto',
                                  linkage='ward',
                                  compute_distances=True)

cluster.fit_predict(vecs)

plot_dendrogram(cluster)