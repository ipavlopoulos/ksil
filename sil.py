from sklearn.metrics import silhouette_samples
import pandas as pd
import numpy as np

def macro(X, clustering, per_cluster=False):
    """
    Macro-averaged silhouette, for per-cluster averaging, then returning their mean
    :return: the silhouette aggregated score
    """
        
    point_sils = pd.DataFrame({'sil': silhouette_samples(X, clustering), 'label':clustering})

    representatives = point_sils.groupby('label').sil.apply(np.mean)
    if per_cluster:
        return representatives
    else:
        return representatives.mean()
    
def micro(X, clustering):
    """
    Micro-averaged silhouette, as in sklearn
    :return: the silhouette aggregated score
    """
        
    point_sils = pd.DataFrame({'sil': silhouette_samples(X, clustering), 'label':clustering})
    return point_sils.sil.mean()

def sample_per_clust(X, clustering, frac=0.5):
    """
    Per-cluster balanced sampling
    :return: data points subsampled so that a balanced clustering is achieved 
    """
    labels_pd = pd.Series(clustering) # a series of the cluster assignments 
    support = labels_pd.value_counts() # the sizes per label (type)
    ssample_size = int(support.max()*frac)
    ssample_size = min(ssample_size, support.min())
    indices = [] # for each label sample data points
    for label in support.index:
        indices.extend(labels_pd[labels_pd==label].sample(ssample_size).index)
    return pd.Series(X)[indices].values, labels_pd[indices].values

def macro_eval(X, clustering, measure):
    X, clustering = sample_per_clust(X, clustering)
    return measure(X, clustering)
