from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import LedoitWolf

class kdf(KernelDensityGraph):

    def __init__(self, feature_sub=1, kwargs={}):
        super().__init__()

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.feature_used = {}
        self.feature_sub = feature_sub
        self.kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y):
        r"""
        Fits the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        """
        if self.is_fitted:
            raise ValueError(
                "Model already fitted!"
            )
            return

        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)
        feature_dim = X.shape[1]
        feature = list(range(feature_dim))
        np.random.shuffle(feature)
        feature_per_group = feature_dim//self.feature_sub
        self.features = []
        
        for ii in range(self.feature_sub-1):
            self.features.append(
                feature[ii*feature_per_group:(ii+1)*feature_per_group]
            )

        self.features.append(
            feature[(self.feature_sub-1)*feature_per_group:]
        )

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            self.feature_used[label] = []

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            _, polytope_idx = np.unique(
                predicted_leaf_ids_across_trees, return_inverse=True, axis=0
            )
            total_polytopes_this_label = np.max(polytope_idx)+1

            for polytope in range(total_polytopes_this_label):
                matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == predicted_leaf_ids_across_trees[polytope],
                    axis=1
                )
                idx = np.where(
                    matched_samples>0
                )[0]

                if len(idx) == 1:
                    continue
                

                scales = matched_samples[idx]/np.max(matched_samples[idx])

                for sub_group in range(self.feature_sub):
                    X_tmp = X_[idx].copy()
                    X_tmp = X_tmp[:,self.features[sub_group]].copy()
                    #print(X_tmp.shape)
                    location_ = np.average(X_tmp, axis=0, weights=scales)
                    X_tmp -= location_
                    
                    sqrt_scales = np.sqrt(scales).reshape(-1,1) @ np.ones(len(self.features[sub_group])).reshape(1,-1)
                    X_tmp *= sqrt_scales
                    #X_tmp[:,-10:-1] *= 1.2
                    
                    covariance_model = LedoitWolf(assume_centered=True)
                    covariance_model.fit(X_tmp)

                    self.polytope_means[label].append(
                        location_
                    )
                    self.polytope_cov[label].append(
                        covariance_model.covariance_*len(idx)/sum(scales)
                    )
                    self.feature_used[label].append(
                        self.features[sub_group]
                    )

            '''self.polytope_mean_cov[label] = np.average(
                    self.polytope_cov[label], 
                    axis= 0,
                    weights = self.polytope_cardinality[label]
                    )'''

        self.is_fitted = True
        
            
    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]
        #polytope_cov = self.polytope_mean_cov[label]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)
        return likelihood

    def predict_proba(self, X):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += np.nan_to_num(self._compute_pdf(X[:,self.feature_used[label][polytope_idx]], label, polytope_idx))

        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-100)).T
        return proba

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        return np.argmax(self.predict_proba(X), axis = 1)