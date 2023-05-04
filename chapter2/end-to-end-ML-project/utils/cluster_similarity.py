from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted, check_array
from sklearn.metrics.pairwise import rbf_kernel

import numpy as np
from typing import Union, List
from pandas import Series

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_init:int=10,
                 random_state:Union[None,int]=None,
                 gamma=.1,
                 n_clusters:int=10) -> None:
        self.n_init = n_init
        self.random_state = random_state
        self.gamma = gamma
        self.n_clusters = n_clusters
    
    def fit(self,
            X: Union[Series, np.array],
            y=None,
            sample_weight:Union[Series, np.array, None]=None,
            ):
        X = check_array(X)
        if sample_weight:
            assert sample_weight.shape[0] == X.shape[0], "'X' and sample_weight' must have the same amount of records"
            sample_weight = check_array(sample_weight)

        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        
        return self

    def transform(self,
                  X: Union[Series, np.array]) -> np.array:
        X = check_array(X)
        check_is_fitted(self)
        return rbf_kernel(
            X,self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None) -> List[str]:
        return [f"Cluster_{i}_similarity" for i in range(self.n_clusters)]