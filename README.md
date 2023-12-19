# The K-Silhouette (k-sil) Algorithm

* It labels the data based on the points with the maximum silhouette score in the cluster.
* All points are assigned a random integer label $k \in \{1, ..., K\}$, forming a partition of the dataset.
* The point with the highest silhouette score per cluster is extracted and used as a cluster representative.
* All the points in the dataset, then, are assigned a label ($k$) according to their closest representative.
* Note that the number of clusters ($K$) can decrease during training, because a cluster can disappeared.

---

After cloning the repository, run the following in Python:
```
>>> from models import ksil
>>> centres, clustering, history = ksil(points=X, k=500, patience=10)
>>> history.plot(); # to visualise the history of the macro-averaged silhouette
```

To assess based on macro-averaged silhouette, run the following:
```
>>> import sil
>>> print(f'Macro-Sil: {sil.macro(X, clustering):.2f}')
```
