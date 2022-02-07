import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kdg.utils import generate_gaussian_parity
from kdg import kdn
from tensorflow import keras
import random


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c


def plot_gaussians(Values, Classes, ax=None):
    X = Values
    y = Classes
    colors = sns.color_palette("Dark2", n_colors=2)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, y), s=50)
    ax.set_title("Created Gaussians", fontsize=30)
    plt.tight_layout()


def label_noise_trial_clf(n_samples, p=0.10, n_estimators=500, clf=None):
    """
    Runs a single trial of the label noise experiment at
    a given contamination level for any given classifiers.

    Parameters
    ---
    n_samples : int
        The number of training samples to be generated
    p : float
        The proportion of flipped training labels
    n_estimators : int
        Number of trees in the KDF and RF forests
    clf : classifier class 

    Returns
    ---
    err : list
        A collection of the estimated error of 
        a given classifier on a test distribution
    """
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.5)
    X_val, y_val = generate_gaussian_parity(n_samples / 2, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    err = []

    fit_kwargs = {
    "epochs": 300,
    "batch_size": 32,
    "verbose": True,
    "validation_data": (X_val, keras.utils.to_categorical(y_val)),
    "callbacks": [callback]}

    mode = ['none', 'lin'][1]

    for i, c in enumerate(clf):
        if i == 3: #nn
            c.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
            err.append((1 - np.mean(np.argmax(c.predict(X_test), axis=1) == y_test)))
            #kdn
            c_kdn = kdn(network=c, weighting_method=mode, T=10, c=1, verbose=False)
            # c_kdn = kdn(network=c, k=1e-4, weighting_method=mode, T=2, c=1, verbose=False)
            c_kdn.fit(X, y)
            err.append(1 - np.mean(c_kdn.predict(X_test) == y_test))
        else:
            c.fit(X, y)
            err.append(1 - np.mean(c.predict(X_test) == y_test))

    #swap orders from [KDF, SVM, RF, NN, KDN, SPORF]
    #to [SVM, RF, NN, SPORF, KDF, KDN]

    err = np.array(err)[[1,2,3,5,0,4]].tolist()

    return err