import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def arg_percentile(data, percentile=75):
    """ Find the index with the respective percentile
    
    :param data: the data
    :param percentile: the percentage of the data one is interested in
    :param index: returns the index 
    :return: the index at the specific percentile
    """
    percentile_value = np.percentile(data, percentile)
    sorted_data = np.sort(data)
    percentile_index = np.searchsorted(sorted_data, percentile_value)
    return percentile_index
    


def silmeans(X, k, ssize=-1, max_iter=1000, patience=20, e=1e-06, init='random', percentile=0.5, warmup=0, seed=2024):
    """Data clustering with silmeans by using the points with the highest silhouette per cluster as centres

    :param points: the data
    :param k: the starting number of clusters, if warmup is on, set it high, otherwise set it to the desired K
    :param ssize: the number of points to sample per cluster to evaluate silhouette, ignored by default
    :param max_iter: maximum number of iterations
    :param patience: number of epochs to wait with no improvement
    :param e: the value below which we assume convergence
    :param init: starting setting (kmeans/random)
    :param percentile: the percent of the data above which the silhouette should be to consider in the centroid computation 
    :param warmup: the number of steps during which k is estimated; by default, 20
    :param seed: random state
    :return: the (best) centres, assigned labels, the history of the macro-score
    """
    
    assert init in {'kmeans', 'random'}
    
    # initializing
    size = len(X)
    stop, best_score, best_centres, best_clustering, kappas = 0, 0, [], [], []    
    results = {'mean':[], 'sem':[], 'k':[]}

    # STEP: START (move inside the loop to estimate K)
    if init == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        clustering = kmeans.labels_
        centres = kmeans.cluster_centers_
    elif init == 'random':
        # Pick K centres, randomly the first time 
        clustering = [np.random.randint(k) for i in range(size)]
        centre_idx = np.random.choice(range(size), k, replace=False)
        centres = X[centre_idx]
    else:
        print('Not implemented yet')
        return best_centres, best_clustering, pd.DataFrame(results)

    for iteration in range(max_iter):         
        
        # STEP: ASSESSMENT        
        # compute one silhouette per point
        data = pd.DataFrame({'clustering': clustering, 'points':X.tolist()})
        data['silhouette'] = silhouette_samples(X=X, labels=clustering)
        
        # group the points per cluster (sorted integers)
        sil_per_cluster = data.groupby('clustering', group_keys=True).silhouette
        
        # compute the macro-averaged silhouette score
        results['mean'].append(sil_per_cluster.apply(np.mean).mean())
        results['sem'].append(sil_per_cluster.apply(np.mean).sem())
        
        # update the best solution
        if results['mean'][-1] > best_score:
            best_score = results['mean'][-1]
            best_clustering = clustering
            best_centres = centres
            stop = 0 # reset
        # or be patient
        else: 
            stop += 1
            
        # STEP: REASSIGNMENT (for the next iteration)        
        if iteration<warmup: # for negative warmpup this will never be activated 
            # first, use just the maximum silhouette to reduce K at an optimum value
            centres = data.iloc[sil_per_cluster.apply(np.argmax).values].points
            # centres = data.iloc[sil_per_cluster.apply(lambda x: arg_percentile(x, percentile=percentile)).values].points
        else:
            # the centre of the points with the highest sil per cluster is the new cluster centre 
            centres = []
            for name, cluster in sil_per_cluster:
                cluster_idx = cluster[cluster>cluster.quantile(percentile)].index
                cluster_points = data.iloc[cluster_idx].points
                w = cluster_points.shape[0]
                if w==0:
                    # a value far far away (effectively reducing K)
                    centres.append(np.sum(centres,axis=0)*1000)
                    # the previous center, an arbitrary choice to enforce the desired K
                    #centres.append(centres[-1])
                else:
                    h = len(cluster_points.iloc[0])
                    centres.append(np.concatenate(cluster_points.to_numpy()).reshape(w,h).mean(0))
            centres = pd.Series(centres)
            
        # assign all the points to their clusters based on the max_sil points
        clustering = [centres.apply(lambda x: np.linalg.norm(x-p)).argmin() for p in X]
        k = len(set(clustering))
        results['k'].append(k)
                        
        # max patience reached
        if stop>patience:
            print(f'Max patience is reached at iteration: {iteration}, for K: {k}, using a solution scored as: {best_score:.3f}')
            return best_centres, best_clustering, pd.DataFrame(results), k

        # converged
        if (np.mean(results['mean'][-patience:]) < e) & (iteration>100):
            print(f'Converged at iteration: {iteration}, for K: {k}, using a solution scored as: {best_score:.3f}')
            return best_centres, best_clustering, pd.DataFrame(results), k
                
    # end of iterations
    print(f'Maximum iterations are reached at iteration: {iteration}, for K: {k}, using a solution scored as: {best_score:.3f}')
    return best_centres, best_clustering, pd.DataFrame(results), k
