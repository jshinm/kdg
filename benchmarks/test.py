#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
import os
import math
# %%
mc_reps = 10
n_estimators = 500

task = openml.tasks.get_task(12)
X, y = task.get_X_and_y()
total_sample = X.shape[0]
unique_classes, counts = np.unique(y, return_counts=True)

test_sample = 100
train_samples = [10, 50, 100]
indx = []

for label in unique_classes:
    indx.append(
        np.where(
            y==label
        )[0]
    )
#%%
err_res = []
err_res_25_quantile = []
err_res_75_quantile = []

err_res_rf = []
err_res_25_quantile_rf = []
err_res_75_quantile_rf = []

for train_sample in train_samples:
    err = []
    err_ = []
    for _ in range(mc_reps):
        indx_to_take_train = []
        indx_to_take_test = []

        for ii, _ in enumerate(unique_classes):
            np.random.shuffle(indx[ii])
            indx_to_take_train.extend(
                list(
                        indx[ii][:train_sample]
                )
            )
            indx_to_take_test.extend(
                list(
                        indx[ii][-test_sample:counts[ii]]
                )
            )
        model_kdf = kdf(feature_sub=2, kwargs={'n_estimators':500})
        model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
        err.append(
            np.mean(
                model_kdf.predict(X[indx_to_take_test])==y[indx_to_take_test]
            )
        )
        err_.append(
            np.mean(
                model_kdf.rf_model.predict(X[indx_to_take_test])==y[indx_to_take_test]
            )
        )

    err_res.append(
        np.median(err)
    )
    err_res_25_quantile.append(
        np.quantile(err,[.25])[0]
    )
    err_res_75_quantile.append(
        np.quantile(err,[.75])[0]
    )

    err_res_rf.append(
        np.median(err_)
    )
    err_res_25_quantile_rf.append(
        np.quantile(err_,[.25])[0]
    )
    err_res_75_quantile_rf.append(
        np.quantile(err_,[.75])[0]
    )
# %%
