#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
reps = 30
n_estimators = range(1,50,2)
n_train = 10000
n_test = 1000
rf_med = []
rf_25_quantile = []
rf_75_quantile = []
# %%
for n_estimator in n_estimators:
    err = []
    for _ in range(reps):
        X, y = sparse_parity(n_train)
        X_test, y_test = sparse_parity(n_test)

        rf_model = rf(n_estimators=n_estimator).fit(X, y)
        err.append(
            1 - np.mean(
                rf_model.predict(X_test)==y_test
            )
        )
    
    rf_med.append(np.median(err))
    rf_25_quantile.append(
            np.quantile(err,[.25])[0]
        )
    rf_75_quantile.append(
            np.quantile(err,[.75])[0]
        )
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(n_estimators, rf_med, c="k", label='RF')
ax.fill_between(n_estimators, rf_25_quantile, rf_75_quantile, facecolor='k', alpha=.3)
ax.set_xticks(range(1,50,4))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
# %%
