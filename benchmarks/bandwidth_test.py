#%%
import numpy as np
from kdg import kdf
from kdg.utils import gaussian_sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
#%%
p = 20
p_star = 3
sample_size = [1000,5000,10000]
n_test = 1000
reps = 10

n_estimators = 50
bws = np.arange(.1,1.1,.1)
df = pd.DataFrame()
sample_list = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []
err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
bandwidth = []

for sample in sample_size:
    X, y = gaussian_sparse_parity(sample, p_star=p_star, p=p)
    X_test, y_test = gaussian_sparse_parity(n_test, p_star=p_star, p=p)

    for bw in bws:
        print('Doing sample ',sample,' for bandwidth ',bw)
        err = []
        err_rf = []
        for _ in range(reps):
            model_kdf = kdf(bw=bw, kwargs={'n_estimators':n_estimators})
            model_kdf.fit(X, y)

            err.append(
                1 - np.mean(model_kdf.predict(X_test)==y_test)
            )
            err_rf.append(
                1 - np.mean(model_kdf.rf_model.predict(X_test)==y_test)
            )
        
        err_kdf_med.append(np.median(err))
        err_kdf_25_quantile.append(
                np.quantile(err,[.25])[0]
            )
        err_kdf_75_quantile.append(
            np.quantile(err,[.75])[0]
        )
        err_rf_med.append(np.median(err_rf))
        err_rf_25_quantile.append(
                np.quantile(err_rf,[.25])[0]
            )
        err_rf_75_quantile.append(
            np.quantile(err_rf,[.75])[0]
        )
        bandwidth.append(bw)
        sample_list.append(sample)

df['kdf err med'] = err_kdf_med
df['kdf err 25 quantile'] = err_kdf_25_quantile
df['kdf err 75 quantile'] = err_kdf_75_quantile
df['rf err med'] = err_rf_med
df['rf err 25 quantile'] = err_rf_25_quantile
df['rf err 75 quantile'] = err_rf_75_quantile
df['sample size'] = sample_list
df['bandwidth'] = bandwidth
df.to_csv('bandwidth_res.csv')
# %%
