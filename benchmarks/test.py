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

# %%
task = openml.tasks.get_task(12)
X, y = task.get_X_and_y()
#X = X[:,100:150]

skf = StratifiedKFold(n_splits=5)
train_index, test_index = list(skf.split(X, y))[0]

model_kdf = kdf(feature_sub=2, kwargs={'n_estimators':500})
model_kdf.fit(X[train_index], y[train_index])

# %%
test_acc = np.mean(model_kdf.predict(X[test_index])==y[test_index])
test_acc2 = np.mean(model_kdf.rf_model.predict(X[test_index])==y[test_index])
# %%
