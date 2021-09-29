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
import matplotlib.pyplot as plt 
import seaborn as sns
# %%
mc_reps = 10
n_estimators = 500

task = openml.tasks.get_task(12)
X, y = task.get_X_and_y()
total_sample = X.shape[0]
unique_classes, counts = np.unique(y, return_counts=True)

test_sample = 100
train_samples = np.array([10, 50, 100])
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
           1 - np.mean(
                model_kdf.predict(X[indx_to_take_test])==y[indx_to_take_test]
            )
        )
        err_.append(
            1 - np.mean(
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
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(train_samples*len(unique_classes), err_res, c="r", label='KDF')
ax.plot(train_samples*len(unique_classes), err_res_rf, c="k", label='RF')

ax.fill_between(train_samples*len(unique_classes), err_res_25_quantile, err_res_75_quantile, facecolor='r', alpha=.3)
ax.fill_between(train_samples*len(unique_classes), err_res_25_quantile_rf, err_res_75_quantile_rf, facecolor='k', alpha=.3)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('Generalization Error')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.legend(frameon=False)

plt.savefig('plots/sim_res.pdf')

# %%
