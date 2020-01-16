print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

import os, sys
sys.path.append(os.path.pardir)
import echo_regression as er


diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
# orthonormalize
_, d, v = np.linalg.svd(X, full_matrices=False)
K = np.sqrt(X.shape[1]) * v.T / d
del v, d
X1 = np.dot(X, K)
X = X1

# Compute paths

eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps, positive=True, fit_intercept=False)
print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

print("Computing regularization path using the positive elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

print("Computing regularization path using Echo...")
alphas_echo, coefs_echo, mis_echo = er.echo_path(X, y, assume_diagonal=True)

# Display results

plt.figure(1)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')
plt.savefig('figures/path1.png')


plt.figure(2)
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

# # Add vertical lines for new nonzero coefficient entering
# ymin, ymax = plt.ylim()
# xx = np.sum(np.abs(coefs_lasso.T), axis=1)
# xx /= xx[-1]
# plt.vlines(xx, ymin, ymax, linestyle='dashed')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')
plt.savefig('figures/path2.png')


plt.figure(3)
neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net and positive Elastic-Net')
plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
           loc='lower left')
plt.axis('tight')
plt.savefig('figures/path3.png')


plt.figure(4)
neg_log_alphas_echo = -np.log10(alphas_echo)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_echo, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_echo, coef_pl, linestyle='--', c=c)

# # Add vertical lines for new nonzero coefficient entering
# ymin, ymax = plt.ylim()
# xx = np.sum(np.abs(coefs_lasso.T), axis=1)
# xx /= xx[-1]
# plt.vlines(xx, ymin, ymax, linestyle='dashed')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Echo')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Echo'), loc='lower left')
plt.axis('tight')
plt.savefig('figures/path4.png')

if '-i' in sys.argv:
    import IPython
    IPython.embed()