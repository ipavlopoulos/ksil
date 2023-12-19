from sklearn.metrics import silhouette_samples

def aggregated(X, clustering, strategy='macro'):
  assert strategy in {'macro', 'micro'}
  point_sils = pd.DataFrame({'sil': silhouette_samples(X, clustering),
                               'label':clustering})
  if strategy == 'micro':
    return point_sils.mean()

  elif strategy == 'macro':
    return point_sils.groupby('label').sil.apply(np.mean)

  else:
    print('Not implemented yet')
    return None
