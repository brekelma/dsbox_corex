""" Echo Regression
Sparse linear regression that bounds the information capacity of the learned model using an Echo compression channel.
Code is in the style of scikit-learn.

Code below by:
Greg Ver Steeg (gregv@isi.edu), 2018.
"""

import numpy as np
from scipy import linalg
from sklearn.linear_model.base import LinearModel, RegressorMixin
from sklearn.utils import check_consistent_length, check_array, check_X_y


def echo_regression(X, y, alpha, solver='svd', assume_diagonal=False):
    """Solve the Echo regression problem.
    Parameters
    ----------
    X : array-like,
        shape = [n_samples, n_features]
        Training data
    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    alpha : float,
        Mutual info regularization strength; must be a positive float.
        Improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
    solver : {"svd"},
        Choose the solver type. Currently only SVD is implemented
    assume_diagonal : bool,
        Whether to assume a diagonal covariance, leading to a simpler solution.
    Returns
    -------
    coef : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).
    Notes
    -----
    This function assumes centered data. Use EchoRegression class for convenience functions.
    """

    _dtype = [np.float64, np.float32]
    X = check_array(X, accept_sparse=False, dtype=_dtype)
    y = check_array(y, dtype=X.dtype, ensure_2d=False)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    n_samples_, n_targets = y.shape

    if n_samples != n_samples_:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_))

    if solver == 'svd':
        coef = _solve_svd(X, y, alpha, assume_diagonal=assume_diagonal)
    else:
        raise ValueError('Solver %s not understood' % solver)

    if ravel:
        coef = coef.ravel()  # When y was passed as a 1d-array, we flatten the coefficients.

    return coef


def _get_solution_basis_svd(X, y, assume_diagonal=False):
    """Given the solution basis, the entire spectrum of solutions over all regularizations can be constructed."""
    if y.ndim == 1:
        y = y[:, np.newaxis]
    b = np.dot(X.T, y) / len(X)  # n samples by n targets   TODO: use svd here too?

    if assume_diagonal:
        b2 = np.square(b)
        b /= b2
        a = b2 / np.var(X, axis=0)[:, np.newaxis]
        return np.eye(len(b)), a, b
    else:
        _, s, U = linalg.svd(X, full_matrices=False)
        s = s**2 / len(X)  # Eigenvalues of covariance matrix
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx][:, np.newaxis]
        U = U[idx]

        UTb = np.dot(U, b)
        UTb2 = np.square(UTb)
        return U.T, UTb2 / s_nnz, UTb / UTb2


def _solve_svd(X, y, alpha, assume_diagonal=False):
    """Analytic solution for Echo regression.
    Assume X is centered, x shape is (n_samples, n_features), y shape is (n_samples, n_targets).
    Returns coefficients
    """
    U, a, b = _get_solution_basis_svd(X, y, assume_diagonal=assume_diagonal)
    coef = U.dot((a - alpha).clip(0.) * b)
    return coef.T  # n outputs by n variables


def echo_path(X, y, assume_diagonal=False):
    """Return coefficient solutions over the entire regularization path
    Returns
    -------
    alphas : array, shape (n_alphas) or (n_alphas, n_outputs)
    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.
    """
    U, a, b = _get_solution_basis_svd(X, y, assume_diagonal=assume_diagonal)
    a_list = np.append(np.ravel(a), 0)
    a_list = np.sort(a_list)
    if y.ndim == 1:
        a = a.ravel()
        b = b.ravel()
    coefs = np.array([(U.dot((a - a_i).clip(0.) * b)).T for a_i in a_list])
    mis = np.array([np.sum(-0.5 * np.log(np.clip(a_i / a, 0, 1)), axis=0) for a_i in a_list])
    return np.sort(a_list), np.moveaxis(coefs, 0, -1), mis


class EchoRegression(LinearModel, RegressorMixin):
    """Least squares regression with information capacity constraint from echo noise.
    Minimizes the objective function::
    E(y - y_hat)^2 + alpha * I(X,y)
    where, X_bar = X + S * echo noise, y_hat = X_bar w + w_0,
    so that I(X,y) <= -log det S,
    with w the learned weights / coefficients.
    The objective simplifies and has an analytic solution.
    Parameters
    ----------
    alpha : float or 'auto'
        Regularization strength; must be a positive float. 'auto' will pick a good value automatically.
    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    assume_diagonal : bool
        Assume that covariance is diagonal, leading to sparsity
        in data basis, instead of covariance eigen-basis
    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    Examples
    --------
    >>> from echo_regression import EchoRegression
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = EchoRegression(alpha=1.0)
    >>> clf.fit(X, y)
    EchoRegression(alpha=1.0, copy_X=True, fit_intercept=True, solver='svd')
    """

    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True, assume_diagonal=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.assume_diagonal = assume_diagonal
        self.mi = None  # Mutual information of solution with given alpha
        self.coef_, self.intercept_ = None, None  # Coefficients of solution

        self.store_basis = True  # TODO: I can't see why people would want this, outside of development
        self.U, self.a, self.b = None, None, None  # Basis

    def fit(self, X, y):
        """Fit Echo Regression Model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        check_consistent_length(X, y)

        if y.ndim > 2:
            raise ValueError("Target y has the wrong shape %s" % str(y.shape))
        ravel = False
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            ravel = True

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=False, copy=self.copy_X)

        U, a, b = _get_solution_basis_svd(X, y, assume_diagonal=self.assume_diagonal)
        if self.alpha == 'auto':
            self.alpha = 0  # TODO
        coef = (U.dot((a - self.alpha).clip(0.) * b)).T  # n outputs by n variables

        if ravel:
            coef = coef.ravel()  # When y was passed as a 1d-array, we flatten the coefficients.

        self.coef_ = coef
        self._set_intercept(X_offset, y_offset, X_scale)
        self.mi = np.sum(-0.5 * np.log(np.clip(self.alpha / a, 0, 1)), axis=0)
        if self.store_basis:
            self.U, self.a, self.b = U, a, b  # Store solution basis, to quickly re-calculate solutions w/ new alpha

        return self

    def produce(self, X):
        result = np.transpose(np.matmul(self.coef_, np.transpose(X)))+self.intercept_
        return result

    def fit_produce(self, X, y):
        model = self.fit(X, y)
        return self.produce(X)




