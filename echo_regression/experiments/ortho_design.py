import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.pardir)
import echo_regression as er

# Generate data
np.random.seed(42)
n_samples, n_features = 300, 100
X = np.random.randn(n_samples, n_features)

beta = 3 * np.random.randn(n_features)
beta[int(0.2 * n_features):] = 0  # sparsify coef
beta = np.sort(beta)  # Sort, for easy visualization
y = X.dot(beta)
y += 0.01 * np.random.normal(size=n_samples)  # Add noise

X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

alpha = 1.

# #############################################################################
# ElasticNet
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

# Lasso
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# Echo
echo = er.EchoRegression(alpha=5 * alpha)
y_pred_echo = echo.fit(X_train, y_train).predict(X_test)
np.set_printoptions(precision=3, suppress=True)
print np.sort(echo.a.ravel())
r2_score_echo = r2_score(y_test, y_pred_echo)
print('echo')
print("r^2 on test data : %f" % r2_score_echo)

# Echo
echo2 = er.EchoRegression(alpha=5 * alpha, assume_diagonal=True)
y_pred_echo = echo2.fit(X_train, y_train).predict(X_test)
print np.sort(echo2.a.ravel())
r2_score_echo2 = r2_score(y_test, y_pred_echo)
print('echo')
print("r^2 on test data : %f" % r2_score_echo2)


#plt.plot(enet.coef_, color='lightgreen', linewidth=2,
#         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(echo.coef_, color='lightgreen', linewidth=2,
         label='Echo coefficients')
plt.plot(echo2.coef_, color='red', linewidth=2,
         label='Echo coefficients (diagonal)')
plt.plot(beta, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Echo R^2: %f"
          % (r2_score_lasso, r2_score_echo))
plt.savefig('figures/coefficients.png')

if '-i' in sys.argv:
    import IPython; IPython.embed()
