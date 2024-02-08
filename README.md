# The silmeans algorithm

* It labels the data based on the centroid of the points with the highest silhouette score in the cluster.
* All points are assigned a random integer label $k \in \{1, ..., K\}$, forming a partition of the dataset.
* The point with the highest silhouette (or the centroid of multiple ones) per cluster is used as the cluster center.
* All the points in the dataset are assigned a label ($k$) according to their closest center.

NOTE: Clusters disappear when using the maximum silhouette as a center, leading to a number of clusters that decreases during training.

---

After cloning the repository, run the following in Python:
```
>>> from models import silmeans
>>> centres, clustering, history = silmeans(points=X, k=500, patience=10)
>>> history.plot(); # to visualise the history of the macro-averaged silhouette
```

To assess based on macro-averaged silhouette, run the following:
```
>>> import sil
>>> print(f'Macro-Sil: {sil.macro(X, clustering):.2f}')
```
