# https://github.com/lmcinnes/umap/blob/master/doc/clustering.rst

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sns.set(style='white', rc={'figure.figsize':(10,8)})

mnist = fetch_mldata('MNIST Original')


standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist.data)

plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist.target, s=0.1, cmap='Spectral');
