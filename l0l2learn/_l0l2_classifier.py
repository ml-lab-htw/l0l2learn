import numpy as np

from mosek.fusion import AccSolutionStatus, Domain, Expr, Matrix, Model, ObjectiveSense, SolutionStatus, Var
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._utils import softplus


class L0L2Classifier(ClassifierMixin, BaseEstimator):
    """L0-constrained L2-regularized logistic regression classifier.

    This estimator solves a mixed-integer conic logistic regression problem
    using MOSEK Fusion. The fitted objective is a softpluss loss plus
    0.5 * lambd * ||coef||_2^2.

    The model uses an L2 penalty controlled by ``lambd``, feature costs c_n,
    and an explicit budget constraint b:

        sum_n c_n z_n <= b

    where z_n indicates whether feature n is selected. By default, all feature
    costs are one, so the budget constraint reduces to a cardinality constraint:

    ||coef||_0 <= b

    Parameters
    ----------
    b : float or None
        Maximum feature selection budget. If ``c=None``, this is equivalent to
        the maximum number of nonzero coefficients. Must be positive. If ``None``,
        no budget constraint is used.

    c : array-like of shape (n_features,), default=None
        Nonnegative feature costs. If ``None``, every feature has cost one.

    lambd : float, default=1.0
        Strength of the L2 penalty. Must be nonnegative.

    fit_intercept : bool, default=True
        Whether to fit an intercept.

    time_limit : float or None, default=None
        Maximum MOSEK solve time in seconds. If ``None``, no time limit is set.

    mosek_log : bool, default=False
        Whether to print MOSEK solver output.

    Attributes
    ----------
    c_ : ndarray of shape (n_features,)
        Feature costs.

    classes_ : ndarray of shape (2,)
        Class labels.

    coef_ : ndarray of shape (1, n_features)
        Fitted coefficients.

    intercept_ : ndarray of shape (1,)
        Fitted intercept.

    sparsity_ : int
        Number of selected features.

    support_ : ndarray of shape (sparsity_,)
        Selected feature indices.
    """

    def __init__(
        self,
        b: float | None = None,
        c: np.ndarray | None = None,
        lambd: float = 1.0,
        fit_intercept: bool = True,
        time_limit: float | None = None,
        mosek_log: bool = False,
    ):
        self.b = b
        self.c = c
        self.lambd = lambd
        self.fit_intercept = fit_intercept
        self.time_limit = time_limit
        self.mosek_log = mosek_log

    def fit(self, X, y):
        """Fit the classifier."""
        X, y = check_X_y(X, y, dtype=np.float64, accept_sparse=False)
        check_classification_targets(y)
        self._validate_params(X)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        if len(encoder.classes_) != 2:
            raise ValueError("L0L2Classifier only supports binary classification.")

        coef, intercept, selected = self._solve(X, y_encoded)

        self.classes_ = encoder.classes_
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=np.float64)
        self.support_ = np.flatnonzero(selected)
        self.sparsity_ = len(self.support_)

        return self

    def decision_function(self, X):
        """Return signed scores for samples.

        Positive scores correspond to ``classes_[1]``.
        Negative scores correspond to ``classes_[0]``.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, accept_sparse=False)

        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X):
        """Return class probabilities."""
        scores = self.decision_function(X)
        p_positive = expit(scores)

        return np.column_stack([1.0 - p_positive, p_positive])

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


    def _solve(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        n, d = X.shape

        with Model("l0l2_classifier") as M:

            if self.time_limit is not None:
                M.setSolverParam("optimizerMaxTime", float(self.time_limit))

            if not self.mosek_log:
                M.setLogHandler(None)
            
            # variables
            theta = M.variable(d)
            theta0 = M.variable(1)
            t = M.variable(n)
            reg = M.variable(d, Domain.greaterThan(0.0))
            z = M.variable(d, Domain.binary())

            # constraints
            if self.b is not None:
                M.constraint(
                    Expr.dot(self.c_.tolist(), z),
                    Domain.lessThan(float(self.b)),
                )

            for j in range(d):
                M.constraint(
                    Var.vstack(z.index(j), reg.index(j), theta.index(j)),
                    Domain.inRotatedQCone(),
                )

            margin = Expr.mul(Matrix.dense(X.tolist()), theta)

            if self.fit_intercept:
                margin = Expr.add(
                    margin,
                    Expr.mul(Matrix.dense([[1.0]] * n), theta0),
                )
            else:
                M.constraint(theta0, Domain.equalsTo(0.0))

            signs = np.where(y == 1, -1.0, 1.0).reshape(-1, 1)

            softplus(
                M,
                t,
                Expr.mulElm(Matrix.dense(signs.tolist()), margin),
            )

            # objective
            M.objective(
                ObjectiveSense.Minimize,
                Expr.add(Expr.sum(t), Expr.mul(float(self.lambd), Expr.sum(reg)))
            )

            M.solve()

            M.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            status = M.getPrimalSolutionStatus()
            if status not in {SolutionStatus.Optimal, SolutionStatus.Feasible}:
                raise RuntimeError(
                    f"MOSEK did not return a primal solution. "
                    f"Primal solution status: {status}; "
                    f"problem status: {M.getProblemStatus()}"
                )

            coef = np.asarray(theta.level(), dtype=np.float64)
            coef_z = np.asarray(z.level(), dtype=np.float64)
            intercept = float(theta0.level()[0])

            selected = coef_z > 0.5
            coef[~selected] = 0.0

        return coef, intercept, selected

    def _validate_params(self, X: np.ndarray) -> None:
        n_features = X.shape[1]

        if self.b is not None:
            if not np.isfinite(self.b):
                raise ValueError("b must be finite or None.")

            if self.b <= 0:
                raise ValueError("b must be positive or None.")

        if self.c is None:
            self.c_ = np.ones(n_features, dtype=np.float64)
        else:
            self.c_ = np.asarray(self.c, dtype=np.float64)

            if self.c_.shape != (n_features,):
                raise ValueError(
                    f"c must have shape (n_features,). "
                    f"Got c.shape={self.c_.shape}, n_features={n_features}."
                )

            if not np.all(np.isfinite(self.c_)):
                raise ValueError("All entries of c must be finite.")

            if np.any(self.c_ < 0):
                raise ValueError("All entries of c must be nonnegative.")

        if not np.isfinite(self.lambd):
            raise ValueError("lambd must be finite.")

        if self.lambd < 0:
            raise ValueError("lambd must be nonnegative.")

        if self.time_limit is not None:
            if not np.isfinite(self.time_limit):
                raise ValueError("time_limit must be finite or None.")

            if self.time_limit <= 0:
                raise ValueError("time_limit must be positive or None.")
