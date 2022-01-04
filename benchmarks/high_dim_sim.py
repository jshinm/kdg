import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kdg import kdf, rkdf
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd

p = 20
p_star = 3
'''sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )'''
sample_size = [1000,5000,10000]
n_test = 1000
reps = 10

n_estimators = 500
df = pd.DataFrame()
reps_list = []
accuracy_kdf = []
# accuracy_kdf_ = []
accuracy_rf = []
# accuracy_rf_ = []
sample_list = []
accuracy_rkdf = []
accuracy_et = []
for sample in sample_size:
    print('Doing sample %d'%sample)
    for ii in range(reps):
        X, y = gaussian_sparse_parity(
            sample,
            p_star=p_star,
            p=p
        )
        X_test, y_test = gaussian_sparse_parity(
            n_test,
            p_star=p_star,
            p=p
        )

        #train kdf
        model_kdf = kdf(n_estimators=n_estimators)

        model_kdf.fit(X, y)
        accuracy_kdf.append(
            np.mean(
                model_kdf.predict(X_test) == y_test
            )
        )
        accuracy_rf.append(
            np.mean(
                model_kdf.rf_model.predict(X_test) == y_test
            )
        )

        # train rkdf
        model_rkdf = rkdf(n_estimators=n_estimators, bootstrap=True, max_features=1)

        model_rkdf.fit(X, y)
        accuracy_rkdf.append(
            np.mean(
                model_rkdf.predict(X_test) == y_test
            )
        )
        accuracy_et.append(
            np.mean(
                model_rkdf.et_model.predict(X_test) == y_test
            )
        )

        reps_list.append(ii)
        sample_list.append(sample)
        # print(accuracy_kdf)
        #train feature selected kdf


df['accuracy kdf'] = accuracy_kdf
df['accuracy rkdf'] = accuracy_rkdf
#df['feature selected kdf'] = accuracy_kdf_
df['accuracy rf'] = accuracy_rf
df['accuracy et'] = accuracy_et
#df['feature selected rf'] = accuracy_rf_
df['reps'] = reps_list
df['sample'] = sample_list

df.to_csv('high_dim_res_kdf_gaussian_random_splits.csv')
# %% plot the result


filename1 = 'high_dim_res_kdf_gaussian_random_splits.csv'

df = pd.read_csv(filename1)

sample_size = [1000,5000,10000]

err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

err_et_med = []
err_et_25_quantile = []
err_et_75_quantile = []

# err_rf_med_ = []
# err_rf_25_quantile_ = []
# err_rf_75_quantile_ = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []

err_rkdf_med = []
err_rkdf_25_quantile = []
err_rkdf_75_quantile = []

# err_kdf_med_ = []
# err_kdf_25_quantile_ = []
# err_kdf_75_quantile_ = []
#clr = ["#e41a1c", "#f781bf", "#306998"]
#c = sns.color_palette(clr, n_colors=3)


for sample in sample_size:
    #err_rf_ = 1 - df['feature selected rf'][df['sample']==sample]
    #err_kdf_ = 1 - df['feature selected kdf'][df['sample']==sample]
    err_rf = 1 - df['accuracy rf'][df['sample']==sample]

    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
    )

    err_et = 1 - df['accuracy et'][df['sample']==sample]

    err_et_med.append(np.median(err_et))
    err_et_25_quantile.append(
            np.quantile(err_et,[.25])[0]
        )
    err_et_75_quantile.append(
        np.quantile(err_et,[.75])[0]
    )

    #err_rf_med_.append(np.median(err_rf_))
    #err_rf_25_quantile_.append(
    #        np.quantile(err_rf_,[.25])[0]
    #    )
    #err_rf_75_quantile_.append(
    #    np.quantile(err_rf_,[.75])[0]
    #)
    err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]

    err_kdf_med.append(np.median(err_kdf))
    err_kdf_25_quantile.append(
            np.quantile(err_kdf,[.25])[0]
        )
    err_kdf_75_quantile.append(
        np.quantile(err_kdf,[.75])[0]
    )

    err_rkdf = 1 - df['accuracy rkdf'][df['sample']==sample]

    err_rkdf_med.append(np.median(err_rkdf))
    err_rkdf_25_quantile.append(
            np.quantile(err_rkdf,[.25])[0]
        )
    err_rkdf_75_quantile.append(
        np.quantile(err_rkdf,[.75])[0]
    )
    #err_kdf_med_.append(np.median(err_kdf_))
    #err_kdf_25_quantile_.append(
    #        np.quantile(err_kdf_,[.25])[0]
    #    )
    #err_kdf_75_quantile_.append(
    #    np.quantile(err_kdf_,[.75])[0]
    #)

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, err_rf_med, c="k", label='RF')
ax.fill_between(sample_size, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(sample_size, err_et_med, c="b", label='ET')
ax.fill_between(sample_size, err_et_25_quantile, err_et_75_quantile, facecolor='b', alpha=.3)

#ax.plot(sample_size, err_rf_med_, c="g", label='RF (feature selected)')
#ax.fill_between(sample_size, err_rf_25_quantile_, err_rf_75_quantile_, facecolor='g', alpha=.3)

ax.plot(sample_size, err_kdf_med, c="r", label='KDF')
ax.fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

ax.plot(sample_size, err_rkdf_med, c="g", label='RKDF')
ax.fill_between(sample_size, err_rkdf_25_quantile, err_rkdf_75_quantile, facecolor='g', alpha=.3)

#ax.plot(sample_size, err_kdf_med_, c="b", label='KDF (feteaure selected)')
#ax.fill_between(sample_size, err_kdf_25_quantile_, err_kdf_75_quantile_, facecolor='b', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/high_dim_gaussian_random_splits.pdf')
