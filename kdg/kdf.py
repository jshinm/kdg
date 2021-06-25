from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdf(KernelDensityGraph):

    def __init__(self, covariance_types = 'full', criterion=None, kwargs={}):
        super().__init__()
        
        if isinstance(covariance_types, str)==False and criterion == None:
            raise ValueError(
                    "The criterion cannot be None when there are more than 1 covariance_types."
                )
            return

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.tree_to_leaf_high = {}
        self.tree_to_leaf_low = {}
        self.kwargs = kwargs
        self.covariance_types = covariance_types
        self.criterion = criterion

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
        low = [-np.inf]*dimension
        high = [np.inf]*dimension

        for ii, tree in enumerate(self.rf_model.estimators_):
            self.tree_to_leaf_high[ii] = {}
            self.tree_to_leaf_low[ii] = {}
            leaf_ids = tree.apply(X)
            leaves_this_tree = np.where(tree.tree_.feature==-2)

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
                        self.tree_to_leaf_high[ii][leaf][jj] = min(X[idx,jj])



        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            total_polytopes_this_label = len(X_)

            for polytope in range(total_polytopes_this_label):

                '''matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == predicted_leaf_ids_across_trees[polytope],
                    axis=1
                )
                idx = np.where(
                    matched_samples>0
                )[0]

                if len(idx) == 1:
                    continue
                
                if self.criterion == None:
                    gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_[idx])
                    self.polytope_means[label].append(
                            gm.means_[0]
                    )
                    self.polytope_cov[label].append(
                            gm.covariances_[0]
                    )
                else:
                    min_val = np.inf
                    tmp_means = np.mean(
                        X_[idx],
                        axis=0
                    )
                    tmp_cov = np.var(
                        X_[idx],
                        axis=0
                    )
                    
                    for cov_type in self.covariance_types:
                        try:
                            gm = GaussianMixture(n_components=1, covariance_type=cov_type, reg_covar=1e-3).fit(X_[idx])
                        except:
                            warnings.warn("Could not fit for cov_type "+cov_type)
                        else:
                            if self.criterion == 'aic':
                                constraint = gm.aic(X_[idx])
                            elif self.criterion == 'bic':
                                constraint = gm.bic(X_[idx])

                            if min_val > constraint:
                                min_val = constraint
                                tmp_cov = gm.covariances_[0]
                                tmp_means = gm.means_[0]
                        
                    self.polytope_means[label].append(
                        tmp_means
                    )
                    self.polytope_cov[label].append(
                        tmp_cov
                    )
        '''
            
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