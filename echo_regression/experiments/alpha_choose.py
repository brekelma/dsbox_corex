import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, lasso_path, enet_path, lars_path
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import os, sys
sys.path.append(os.path.pardir)
import echo_regression as er
plt.style.use('/Users/gregv/Dropbox/Public/gv.mplstyle')


t10 = '#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'.split(',')  # colors

def make_regression(n_samples=100, n_features=50, effective_rank=10, tail_strength=0.5):
    """Make a synthetic regression problem using low rank matrices with an eigenspectrum of some
    effective rank with a tail. Splits matrix to produce train and test set.
    """
    X0 = make_low_rank_matrix(n_samples=2 * n_samples, n_features=n_features + 1,
                              effective_rank=effective_rank, tail_strength=tail_strength)
    X0 -= np.sum(X0, axis=0)
    X_train, X_test = X0[:n_samples, :n_features], X0[n_samples:, :n_features]
    y_train, y_test = X0[:n_samples, n_features], X0[n_samples:, n_features]
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = make_regression(n_samples=400, n_features=500, effective_rank=10, tail_strength=0.5)
alphas, coefs, mis = er.echo_path(X_train, y_train)

test_scores = [np.mean(np.square(y_test - X_test.dot(coef))) for coef in coefs.T]
train_scores = [np.mean(np.square(y_train - X_train.dot(coef))) for coef in coefs.T]
f, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(10,3))
ax1.loglog(alphas[1:-1], test_scores[1:-1], marker='.', label='Test')
ax1.loglog(alphas[1:-1], train_scores[1:-1], marker='.', label='Train')

e_model = er.EchoRegression().fit(X_train, y_train)
gams = np.sort(e_model.a[:,0])
objective = [np.mean(y_train**2) - np.sum((gams - gam).clip(0) + 0.5 * gam * np.log(np.clip(gam / gams, 0, 1)))  for gam in gams]
print 'obj', objective
print len(gams), gams
ax3.loglog(gams, objective, marker='.', label='Objective')
ax3.axvline(gams[np.argmin(objective)])

print 'alpha', alphas
print 'test', test_scores

ax2.plot(mis[1:-1], test_scores[1:-1], marker='.', label='Test')
ax2.plot(mis[1:-1], train_scores[1:-1], marker='.', label='Train')

dhy = (np.array(train_scores[1:]) - np.array(train_scores[:-1])) / (alphas[1:] - alphas[:-1])
dhy *= 0.7 * np.min(test_scores) / np.max(dhy)
ax2.plot(mis[1:], dhy, marker='*', label='$\Delta h(y)$')

mis = mis[1:-1]
mis = mis[::-1]
mis_per = mis / np.arange(1, len(mis) + 1)
mis_per *= 0.8 * np.min(test_scores) / np.max(mis_per)
ax2.plot(mis, mis_per, marker='x', label='MI/#coef')

ax1.set_ylabel('Mean Squared Error')
ax1.set_xlabel(r'$\alpha$')
ax3.set_ylabel('Mean Squared Error')
ax3.set_xlabel(r'$\alpha$')
ax2.set_xlabel('$I(X;Y)$')
ax1.legend(loc="best")
ax2.legend(loc="best")
ax3.legend(loc="best")
# ax2.axvline(np.log(len(X_train)), linestyle='--', color='.5')
# plt.xlim([alphas[0], alphas[-1]])
# ax1.set_ylim(0, 1.5 * np.min(test_scores))

plt.savefig('figures/alpha_choose.png')

f, ax3 = plt.subplots(1, 1)

# compare lasso, ridge
# LASSO
eps = np.min(alphas[1:-1]) / np.max(alphas[1:-1])  # the smaller it is the longer is the path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps, fit_intercept=False)
test_scores_lasso = [np.mean(np.square(y_test - X_test.dot(coef))) for coef in coefs_lasso.T]

# E net
alphas_enet, coefs_enet, _ = enet_path(X_train, y_train, eps=eps, l1_ratio=0.5, fit_intercept=False)
test_scores_enet = [np.mean(np.square(y_test - X_test.dot(coef))) for coef in coefs_enet.T]

# Ridge
alphas_ridge, coefs_ridge, _ = enet_path(X_train, y_train, eps=eps, l1_ratio=0., fit_intercept=False, alphas=alphas_enet)
test_scores_ridge = [np.mean(np.square(y_test - X_test.dot(coef))) for coef in coefs_ridge.T]

# Plot
ax3.loglog(alphas[1:-1] / np.max(alphas[1:-1]), test_scores[1:-1], label='Echo',  color=t10[0])
ax3.loglog(alphas_lasso / np.max(alphas_lasso), test_scores_lasso, label='LASSO', color=t10[1])
ax3.loglog(alphas_ridge / np.max(alphas_ridge), test_scores_ridge, label='Ridge', color=t10[2])
ax3.loglog(alphas_enet / np.max(alphas_enet), test_scores_enet, label='Elastic net', color=t10[3])

best_echo, best_lasso, best_ridge, best_enet = map(np.min, [test_scores, test_scores_lasso, test_scores_ridge, test_scores_enet])
ax3.axhline(best_echo, linestyle=':', color=t10[0])
ax3.axhline(best_lasso, linestyle=':', color=t10[1])
ax3.axhline(best_ridge, linestyle=':', color=t10[2])
ax3.axhline(best_enet, linestyle=':', color=t10[3])
print(alphas_lasso)
print("echo:{}, lasso: {}, ridge: {}, none: {}".format(best_echo, best_lasso, best_ridge, test_scores_lasso[-1]))

ax3.set_ylim(min(best_echo, best_lasso, best_ridge, best_enet) * 0.95, test_scores_lasso[-1] * 10)

ax3.set_ylabel('Mean Squared Error')
ax3.set_xlabel(r'$\alpha / \alpha_{max}$')
ax3.legend(loc="best")
plt.savefig('figures/compare_test.png')

if '-i' in sys.argv:
    import IPython
    IPython.embed()
