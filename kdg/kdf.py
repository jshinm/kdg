from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdf(KernelDensityGraph):

    def __init__(self, kwargs={}):
        super().__init__()

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_mean_cov = {}
        self.tree_to_leaf_high = {}
        self.tree_to_leaf_low = {}
        self.kwargs = kwargs

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
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)
        dimension = X.shape[1]

        #profile the leaves
        low = np.array([-np.inf]*dimension)
        high = np.array([np.inf]*dimension)

        for ii, tree in enumerate(self.rf_model.estimators_):
            self.tree_to_leaf_high[ii] = {}
            self.tree_to_leaf_low[ii] = {}
            leaf_ids = tree.apply(X)
            leaves_this_tree = np.where(tree.tree_.feature==-2)[0]
            
            def profile_leaf(node, low_, high_):
                high_tmp1 = high_.copy()
                low_tmp2 = low_.copy()
                feature_in_use = tree.tree_.feature[node]

                if feature_in_use == -2:
                    self.tree_to_leaf_high[ii][node] = high_tmp1
                    self.tree_to_leaf_low[ii][node] = low_tmp2
                    return 

                high_tmp1[feature_in_use] = tree.tree_.threshold[node]
                low_tmp2[feature_in_use] = tree.tree_.threshold[node]

                profile_leaf(tree.tree_.children_left[node], low_.copy(), high_tmp1)
                profile_leaf(tree.tree_.children_right[node], low_tmp2, high_.copy())

            profile_leaf(0, low, high) 

            for leaf in leaves_this_tree:
                idx = np.where(leaf_ids==leaf)[0]

                for jj in range(dimension):
                    if self.tree_to_leaf_high[ii][leaf][jj] == np.inf:
                        self.tree_to_leaf_high[ii][leaf][jj] = max(X[idx,jj])

                    if self.tree_to_leaf_low[ii][leaf][jj] == -np.inf:
                        self.tree_to_leaf_low[ii][leaf][jj] = min(X[idx,jj])


        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            total_polytopes_this_label = len(X_)

            for polytope in range(total_polytopes_this_label):
                polytope_leaves_across_trees = predicted_leaf_ids_across_trees[polytope]

                for ii, leaf in enumerate(polytope_leaves_across_trees):
                    bounds_high = self.tree_to_leaf_high[ii][leaf]
                    bounds_low = self.tree_to_leaf_low[ii][leaf]

                    polytope_center = (bounds_high + bounds_low)/2
                    polytope_cov = np.abs( (bounds_high - bounds_low) )/10

                    self.polytope_means[label].append(
                            polytope_center
                        )
                    self.polytope_cov[label].append(
                            np.eye(dimension)*polytope_cov
                        )
            
    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]

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
                likelihoods[:,ii] += np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

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