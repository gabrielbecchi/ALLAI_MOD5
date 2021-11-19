import pandas as p
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
warnings.filterwarnings("ignore")

data = p.read_csv('credit_card.csv')

# PREPROCESSING
del data['CUST_ID']
#data = data.dropna(axis=0, how='any')
data = data.fillna(data.mean())

# CLUSTERING
for n_clusters in range(2,3):
	# KNN
	clusterer = KMeans(n_clusters=n_clusters)
	preds = clusterer.fit_predict(data)
	score = silhouette_score(data, preds)
	print("KMeans........: n_clusters = {}, silhouette score = {:.4f}".format(n_clusters, score))
	#visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
	#visualizer.fit(data)
	#visualizer.show()

	# MINI BATCH
	clusterer = MiniBatchKMeans(n_clusters=n_clusters)
	preds = clusterer.fit_predict(data)
	score = silhouette_score(data, preds)
	print("MiniBatch.....: n_clusters = {}, silhouette score = {:.4f}".format(n_clusters, score))
	#visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
	#visualizer.fit(data)
	#visualizer.show()

	# SPECTRAL CLUSTERING
	'''
	clusterer = SpectralClustering(n_clusters=n_clusters)
	preds = clusterer.fit_predict(data)
	score = silhouette_score(data, preds)
	print("Spectral Clst.: n_clusters = {}, silhouette score = {:.4f}".format(n_clusters, score))
	#visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
	#visualizer.fit(data)
	#visualizer.show()
	'''

	# hierarchical CLUSTERING
	clusterer = AgglomerativeClustering(n_clusters=n_clusters)
	preds = clusterer.fit_predict(data)
	score = silhouette_score(data, preds)
	print("Hierarchical..: n_clusters = {}, silhouette score = {:.4f}".format(n_clusters, score))
	#visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
	#visualizer.fit(data)
	#visualizer.show()

# DBSCAN

clusterer = DBSCAN(eps=5,min_samples=2)
preds = clusterer.fit_predict(data)
n_clusters = len(np.unique(preds))
score = silhouette_score(data, preds)
print("DBSCAN........: n_clusters = {}, silhouette score = {:.4f}".format(n_clusters, score))
#visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
#visualizer.fit(data)
#visualizer.show()


