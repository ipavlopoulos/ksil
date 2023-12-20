import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples


def ksil(points, k, ssize=-1, max_iter=1000, patience=20, e=1e-06):
    """K-Silhouette clustering of data by using the points with the maximum silhouette per cluster as centres

    :param points: the data
    :param k: the number of clusters
    :param ssize: the number of points to sample per cluster to evaluate silhouette, ignored by default
    :param max_iter: maximum number of iterations
    :param patience: number of epochs to wait with no improvement
    :param e: the value below which we assume convergence
    :return: the (best) centres, assigned labels, the history of the macro-score
    """
    
    size = len(points)
    stop, best_score, best_centres, best_clustering = 0, 0, [], []

    # Pick K centres at random
    centre_idx = np.random.choice(range(size), k, replace=False)
    centres = points[centre_idx]

    # The list of labels, random at start
    clustering = [np.random.randint(k) for i in range(size)]
    
    results = {'mean':[], 'sem':[]}
    
    # Define a convergence criterio and start
    for iteration in range(max_iter): 

        data = pd.DataFrame({'clustering': clustering, 
                             'points':points.tolist()})

        if ssize>0:
            # sample per cluster assignment (use the clustering of the previous iteration)
            cluster_samples = [] 
            for k_iter in range(k):
                cluster_points = data[data.clustering==k_iter] # fetch points in the k_iter cluster
                num_points = min(cluster_points.shape[0], ssize) # up to the min sample)size
                cluster_samples.append(cluster_points.sample(num_points))
            data = pd.concat(cluster_samples) # to dataframe
        
        # computing silhouette on the balanced space
        data_matrix = np.concatenate(data.points.apply(np.array).to_numpy()).reshape(data.shape[0], points.shape[1])
        data['silhouette'] = silhouette_samples(X=data_matrix, labels=data.clustering.values)
        
        # group all points per (previously-assigned) cluster (==> sorted)
        sil_per_cluster = data.groupby('clustering').silhouette
        
        # use the points with the max sil as the new centres
        centres = data.iloc[sil_per_cluster.apply(np.argmax).values].points
        
        # assign all the points to their clusters based on the max_sil points
        new_assignments = [centres.apply(lambda x: np.linalg.norm(x-p)).argmin() for p in points]

        # proceed iff the solution makes sense
        if len(set(new_assignments)) > 1:
            
            # label re-assignment
            clustering = new_assignments                 
            
            # compute the macro-averaged silhouette score
            results['mean'].append(sil_per_cluster.apply(np.mean).mean())
            results['sem'].append(sil_per_cluster.apply(np.mean).sem())
                           
            # update a stopping criterion
            if results['mean'][-1] > best_score:
                best_score = results['mean'][-1]
                best_clustering = clustering
                best_centres = centres
                stop = 0 # reset
            else:
                stop+=1

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
