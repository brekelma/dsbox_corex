import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.linear_model import Ridge, Lasso

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
    print("Method: {}\tBest MSE: {:.2f}\tAt: {}".
          format(method_name, np.min(test_scores_mean), param_range[np.argmin(test_scores_mean)]))
    # plt.ylim(0.0, 1.1)
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
    plt.savefig('figures/best_alpha_{}.png'.format(method_name))
    plt.clf()

np.random.seed(0)

# digits = load_digits()
# X, y = digits.data, digits.target
#
data = load_diabetes()
print('original shape:', data.data.shape, data.target.shape)
X, y = data.data[:], data.target[:]
print('data shape:', X.shape)
# X, y = make_regression()

methods = [("Ridge", Ridge(), "alpha", np.logspace(-3, 0, 20)),
           ("LASSO", Lasso(), "alpha", np.logspace(-3, 0, 20)),
           ("Echo", er.EchoRegression(), 'alpha', np.logspace(-1, 4, 20))]

for method_name, method, param_name, param_range in methods:
    train_scores, test_scores = validation_curve(method, X, y, cv=10, scoring='neg_mean_squared_error',
                                                 param_name=param_name,
                                                 param_range=param_range)
    train_scores *= -1
    test_scores *= -1
    alpha_plot(train_scores, test_scores, param_range, method_name)
