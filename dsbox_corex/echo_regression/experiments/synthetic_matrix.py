import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import validation_curve
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import Ridge, Lasso, LinearRegression

import os, sys
sys.path.append(os.path.pardir)
import echo_regression as er


def alpha_plot(train_scores, test_scores, param_range, method_name):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with {}".format(method_name))
    plt.xlabel(r"$\alpha$")
    plt.ylabel("MSE")
    print("Method: {}\tBest MSE: {:.2e}\tAt: {}".
          format(method_name, np.min(test_scores_mean), param_range[np.argmin(test_scores_mean)]))
    # plt.ylim(0.0, 1.1)
    if len(param_range) > 1:
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig('figures/syn_best_alpha_{}.png'.format(method_name))
        plt.clf()

np.random.seed(0)
n_samples = 200
n_features = 500
e_rank = 30
X = make_low_rank_matrix(n_samples=n_samples,
                         n_features=n_features + 1,
                         effective_rank=e_rank,
                         tail_strength=0.5)
X, y = X[:,:-1], X[:,-1]
print('data shape:', X.shape)



methods = [("Ridge", Ridge(), "alpha", np.logspace(-3, 0, 20)),
           ("LASSO", Lasso(), "alpha", np.logspace(-3, 0, 20)),
           ("Echo", er.EchoRegression(), 'alpha', np.logspace(-1, 4, 20)),
           ("OLS", LinearRegression(), "fit_intercept", [True])]

for method_name, method, param_name, param_range in methods:
    train_scores, test_scores = validation_curve(method, X, y, cv=10, scoring='neg_mean_squared_error',
                                                 param_name=param_name,
                                                 param_range=param_range)
    train_scores *= -1
    test_scores *= -1
    alpha_plot(train_scores, test_scores, param_range, method_name)
