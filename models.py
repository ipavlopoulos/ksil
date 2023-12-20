import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def ksil(X, k, ssize=-1, max_iter=1000, patience=20, e=1e-06, init='random'):
    """K-Silhouette clustering of data by using the points with the maximum silhouette per cluster as centres

    :param points: the data
    :param k: the number of clusters
    :param ssize: the number of points to sample per cluster to evaluate silhouette, ignored by default
    :param max_iter: maximum number of iterations
    :param patience: number of epochs to wait with no improvement
    :param e: the value below which we assume convergence
    :param init: starting setting (kmeans/random)
    :return: the (best) centres, assigned labels, the history of the macro-score
    """
    
    assert init in {'kmeans', 'random'}
    
    # initializing
    size = len(X)
    stop, best_score, best_centres, best_clustering = 0, 0, [], []    
    results = {'mean':[], 'sem':[]}

    # starting point
    if init == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        clustering = kmeans.labels_
    elif init == 'random':
        clustering = [np.random.randint(k) for i in range(size)]
    else:
        print('Not implemented yet')
        return best_centres, best_clustering, pd.DataFrame(results)

    # Pick K centres, randomly the first time 
    centre_idx = np.random.choice(range(size), k, replace=False)
    centres = X[centre_idx]

    # 
    for iteration in range(max_iter): 

        # STEP: ASSESSMENT
        
        # compute one silhouette per point
        data = pd.DataFrame({'clustering': clustering, 'points':X.tolist()})
        data['silhouette'] = silhouette_samples(X=X, labels=clustering)
        
        # group the points per cluster (sorted integers)
        sil_per_cluster = data.groupby('clustering').silhouette
        
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
            
        # STEP: REASSIGNMENT
        
        # use the points with the max sil as the new centres
        centres = data.iloc[sil_per_cluster.apply(np.argmax).values].points
                        
        # assign all the points to their clusters based on the max_sil points
        new_assignments = [centres.apply(lambda x: np.linalg.norm(x-p)).argmin() for p in X]

        # iff the solution makes sense update the clustering
        if len(set(new_assignments)) > 1:                                                   
            # label re-assignment
            clustering = new_assignments                 
                
        # max patience reached
        if stop>patience:
            print(f'Max patience is reached at K={len(set(clustering))}, using the solution with score {best_score:.2f}')
            return best_centres, best_clustering, pd.DataFrame(results)

        # converged
        if (np.mean(results['mean'][-patience:]) < e) & (iteration>100):
            print(f'Converged at K={len(set(clustering))} at a solution with sil: {best_score:.2f}')
            return best_centres, best_clustering, pd.DataFrame(results)
                
    # end of iterations
    print(f'Maximum iterations are reached K={len(set(clustering))}, using the solution with score {best_score:.2f}')
    return best_centres, best_clustering, pd.DataFrame(results)
