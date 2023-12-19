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
    return point_sils.mean()
