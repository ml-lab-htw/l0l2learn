import numpy as np
import pandas as pd
import time
import warnings

from collections import Counter
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state, resample
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted
from tqdm.auto import tqdm

from ._l0l2_classifier import L0L2Classifier
from ._utils import make_preprocessor, resolve_feature_names, resolve_subsample_size


class ResampledL0L2Classifier(ClassifierMixin, BaseEstimator):
    """Resampled L0-constrained L2-regularized logistic regression classifier.

    This estimator repeatedly fits ``L0L2Classifier`` on resampled rows,
    columns, or both. It then aggregates selected feature sets using either
    model selection frequencies or variable inclusion frequencies.

    Parameters
    ----------
    b : float
        Maximum feature selection budget. If ``c=None``, this is equivalent to
        the maximum number of nonzero coefficients. Must be positive.

    c : array-like of shape (n_features,), default=None
        Nonnegative feature costs. If ``None``, every feature has cost one.

    resampling : {"auto", "bootstrap_rows", "subsample_rows", "subsample_columns",
                  "subsample_both", "none"}, default="auto"
        Resampling strategy.

        ``"auto"`` uses row bootstrapping if ``n_samples <= 10000`` and row
        subsampling otherwise. It uses all columns if ``n_features <= 1000``
        and column subsampling otherwise.

    n_row_subsamples : int or float, default=10000
        Number of rows used for row subsampling. If float in ``(0, 1]``, it is
        interpreted as a fraction of rows.

    n_column_subsamples : int or float, default=1000
        Number of columns used for column subsampling. If float in ``(0, 1]``,
        it is interpreted as a fraction of columns.

    n_resamples : int, default=399
        Number of resampling iterations.

    aggregation : {"MSF", "VIF"}, default="MSF"
        Aggregation method.

        ``"MSF"`` selects features based on model selection frequencies.

        ``"VIF"`` selects features whose variable inclusion frequencies exceed
        ``vif_threshold``.

    vif_threshold : float, default=0.5
        Variable inclusion frequency threshold used when ``aggregation="VIF"``.
        Features with inclusion frequencies of at least this value are selected.

    estimator : estimator object, default=None
        Base estimator. If ``None``, ``L0L2Classifier`` is used.

    param_grid : dict or list of dicts, default=None
        Parameter grid passed to ``GridSearchCV``. If ``None``, a default grid
        {"lambd": np.logspace(1, -3, 9)} is used.

    cv : int, cross-validation generator or iterable, default=None
        Cross-validation strategy passed to ``GridSearchCV``. If ``None``,
        a repeated stratified 3-fold CV with 10 repeats is used.

    scoring : str, callable, list, tuple or dict, default="neg_log_loss"
        Scoring passed to ``GridSearchCV``.

    numerical_features : array-like, callable or None, default=None
        Numerical column selector. Integer selectors are interpreted
        positionally against the current DataFrame; other values are interpreted
        as column names. To select an integer-valued column name, use a callable
        selector. If ``None``, all non-categorical columns are numerical.
        Numerical features are imputed and scaled.

    categorical_features : array-like, callable or None, default=None
        Categorical column selector. Integer selectors are interpreted positionally
        against the current DataFrame; other values are interpreted as column
        names. To select an integer-valued column name, use a callable selector.
        If ``None``, no columns are categorical. Categorical features are
        ordinally encoded.

    fit_intercept : bool, default=True
        Whether the underlying ``L0L2Classifier`` fits an intercept.

    mosek_time_limit : float or None, default=None
        Maximum MOSEK solve time in seconds. If ``None``, no time limit is set.

    total_time_limit : float or None, default=None
        Maximum wall-clock time in seconds for the whole resampling procedure.
        If ``None``, no time limit is set.

    max_consecutive_failures : int, default=10
        Maximum number of consecutive failed resampled fits allowed before
        aborting the resampling procedure.

    mosek_log : bool, default=False
        Whether to print MOSEK solver output.

    n_jobs : int or None, default=None
        Number of jobs used to fit resampled models in parallel.
        ``None`` means 1 and ``-1`` means using all available processors.
        Parallelism is applied across resampling iterations.

    random_state : int, RandomState instance or None, default=None
        Random state used for resampling.

    Attributes
    ----------
    c_ : pandas.Series of shape (n_features,)
        Feature costs.

    classes_ : ndarray of shape (2,)
        Class labels.

    coef_ : ndarray of shape (1, n_features)
        Final fitted coefficients after preprocessing.

    intercept_ : ndarray of shape (1,)
        Final fitted intercept after preprocessing.
    
    feature_names_in_ : ndarray of shape (n_features,)
        Feature names.
    
    numerical_features_ : list
        Numerical feature names.

    categorical_features_ : list
        Categorical feature names.

    sparsity_ : int
        Final number of selected features.

    support_ : pandas.Index of shape (sparsity_,)
        Final selected feature names.

    final_estimator_ : estimator
        Final estimator.

    n_resamples_completed_ : int
        Number of resampling iterations that were successfully completed.
        This may be smaller than ``n_resamples`` if ``total_time_limit``
        was reached.

    variable_inclusion_counts_ : pandas.Series of shape (n_features,)
        Number of times each feature was selected.

    variable_availability_counts_ : pandas.Series of shape (n_features,)
        Number of times each feature was available during resampling.

    variable_inclusion_frequencies_ : pandas.Series of shape (n_features,)
        Variable inclusion frequencies.

    model_selection_frequencies_ : dict
        Mapping from selected feature tuples to selection frequencies.

    elapsed_time_ : float
        Elapsed wall-clock time in seconds.
    """

    def __init__(
        self,
        b: float,
        c: np.ndarray | None = None,
        resampling: str = "auto",
        n_row_subsamples: int | float = 10000,
        n_column_subsamples: int | float = 1000,
        n_resamples: int = 399,
        aggregation: str = "MSF",
        vif_threshold: float = 0.5,
        estimator=None,
        param_grid=None,
        cv=None,
        scoring="neg_log_loss",
        numerical_features=None,
        categorical_features=None,
        fit_intercept: bool = True,
        mosek_time_limit: float | None = None,
        total_time_limit: float | None = None,
        max_consecutive_failures: int = 10,
        mosek_log: bool = False,
        n_jobs: int | None = None,
        random_state=None,
    ):
        self.b = b
        self.c = c
        self.resampling = resampling
        self.n_row_subsamples = n_row_subsamples
        self.n_column_subsamples = n_column_subsamples
        self.n_resamples = n_resamples
        self.aggregation = aggregation
        self.vif_threshold = vif_threshold
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.fit_intercept = fit_intercept
        self.mosek_time_limit = mosek_time_limit
        self.total_time_limit = total_time_limit
        self.max_consecutive_failures = max_consecutive_failures
        self.mosek_log = mosek_log
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the resampled classifier."""
        X, y = self._validate_X_y(X, y)
        self._validate_params(X, y)

        self.categorical_features_ = resolve_feature_names(
            X,
            self.categorical_features,
            default=[],
        )

        categorical_set = set(self.categorical_features_)

        self.numerical_features_ = resolve_feature_names(
            X,
            self.numerical_features,
            default=[col for col in X.columns if col not in categorical_set],
        )

        rng = check_random_state(self.random_state)
        seeds = rng.randint(
            np.iinfo(np.int32).max,
            size=self.n_resamples * self.max_consecutive_failures,
        )

        n_samples, n_features = X.shape
        variable_inclusion_counts = pd.Series(0, index=X.columns, dtype=int)
        variable_availability_counts = pd.Series(0, index=X.columns, dtype=int)
        selected_models = []

        parallel = Parallel(
            n_jobs=self.n_jobs,
            return_as="generator",
        )

        jobs = (
            delayed(self._fit_one_resample)(seed, X, y, n_samples, n_features)
            for seed in seeds
        )

        consecutive_failures = 0

        start_time = time.monotonic()

        with tqdm(total=self.n_resamples, desc="Resampling") as pbar:

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*tasks which were still being processed by the workers have been cancelled.*",
                    category=UserWarning,
                )

                for success, col_names, global_support, error_message in parallel(jobs):
                    if len(selected_models) >= self.n_resamples:
                        break

                    if not success:
                        consecutive_failures += 1

                        if consecutive_failures >= self.max_consecutive_failures:
                            raise RuntimeError(
                                f"{consecutive_failures} consecutive resamples failed. "
                                f"Last error was: {error_message}. "
                                "Aborting resampling procedure."
                            )

                        if self.total_time_limit is not None:
                            self.elapsed_time_ = time.monotonic() - start_time
                            if self.elapsed_time_ >= self.total_time_limit:
                                break

                        continue

                    variable_availability_counts.loc[list(col_names)] += 1
                    if len(global_support) > 0:
                        variable_inclusion_counts.loc[list(global_support)] += 1
                    selected_models.append(tuple(global_support))

                    consecutive_failures = 0
                    pbar.update(1)

                    if self.total_time_limit is not None:
                        self.elapsed_time_ = time.monotonic() - start_time
                        if self.elapsed_time_ >= self.total_time_limit:
                            break

        if len(selected_models) == 0:
            raise TimeoutError(
                "total_time_limit was reached before any resampled model was fitted."
            )

        self.variable_inclusion_counts_ = variable_inclusion_counts
        self.variable_availability_counts_ = variable_availability_counts
        self.variable_inclusion_frequencies_ = variable_inclusion_counts.astype(float).divide(
            variable_availability_counts.replace(0, np.nan)
        ).fillna(0.0)

        counts = Counter(selected_models)
        self.n_resamples_completed_ = len(selected_models)
        self.model_selection_frequencies_ = {
            model: count / self.n_resamples_completed_ for model, count in counts.items()
        }

        aggregate_support = self._aggregate_support(counts)
        self.final_estimator_ = self._fit_final_estimator(
            X,
            y,
            aggregate_support,
        )
        self.support_ = pd.Index(aggregate_support)
        self.sparsity_ = len(self.support_)
        self.classes_ = self.final_estimator_.classes_
        self.intercept_ = self.final_estimator_.intercept_
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.coef_ = np.zeros((1, n_features), dtype=np.float64)
        support_positions = X.columns.get_indexer(self.support_)
        self.coef_[:, support_positions] = self.final_estimator_.coef_.reshape(1, -1)

        self.elapsed_time_ = time.monotonic() - start_time

        return self

    def decision_function(self, X):
        """Return signed scores for samples."""
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        X_selected = X.loc[:, self.support_]
        X_preprocessed = self._preprocess(X_selected)

        return self.final_estimator_.decision_function(X_preprocessed)

    def predict_proba(self, X):
        """Return class probabilities."""
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        X_selected = X.loc[:, self.support_]
        X_preprocessed = self._preprocess(X_selected)

        return self.final_estimator_.predict_proba(X_preprocessed)

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        X_selected = X.loc[:, self.support_]
        X_preprocessed = self._preprocess(X_selected)

        return self.final_estimator_.predict(X_preprocessed)


    def _aggregate_support(self, model_counts: Counter) -> np.ndarray:
        if self.aggregation == "MSF":
            selected, _ = model_counts.most_common(1)[0]
            return pd.Index(selected)

        mask = self.variable_inclusion_frequencies_ >= self.vif_threshold
        vif = self.variable_inclusion_frequencies_[mask]

        # greedy budget-feasible selection by decreasing VIF, reduces to
        # top-k filtering when all feature costs are 1
        return self._top_vif_within_budget(vif)

    def _fit_estimator(self, X, y, b, c):
        estimator = self._make_base_estimator(b=b, c=c)

        param_grid = (
            {"lambd": np.logspace(1, -3, 9)}
            if self.param_grid is None
            else self.param_grid
        )

        grid = ParameterGrid(param_grid)
        if len(grid) == 1:
            estimator.set_params(**grid[0])
            estimator.fit(X, y)
            return estimator

        cv = self.cv
        if cv is None:
            _, counts = np.unique(y, return_counts=True)
            min_class_count = counts.min()

            if min_class_count < 3:
                raise ValueError(
                    "Each class needs at least three observations for cross-validation."
                )

            cv = RepeatedStratifiedKFold(
                n_splits=3,
                n_repeats=10,
                random_state=self.random_state,
            )

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            error_score="raise",
            n_jobs=1,
        )
        search.fit(X, y)

        return search.best_estimator_

    def _fit_final_estimator(self, X, y, support):
        support = pd.Index(support)
        if len(support) == 0:
            raise ValueError(
                "Aggregation selected no features. Try increasing b, using MSF, "
                "or lowering the VIF threshold."
            )

        X_selected = X.loc[:, support]
        c_selected = self.c_.loc[support].to_numpy(dtype=np.float64)

        X_preprocessed = self._preprocess(X_selected, fit=True)

        # unconstrained fit on filtered features
        return self._fit_estimator(
            X_preprocessed,
            y,
            b=None,
            c=c_selected,
        )

    def _fit_one_resample(self, seed, X, y, n_samples, n_features):
        try:
            rng = check_random_state(seed)

            row_idx = self._sample_rows(rng, y, n_samples, n_features)
            col_idx = self._sample_columns(rng, n_samples, n_features)
            col_names = X.columns.take(col_idx)

            X_resampled = X.iloc[row_idx].loc[:, col_names]
            y_resampled = y[row_idx]
            c_resampled = self.c_.loc[col_names].to_numpy(dtype=np.float64)

            preprocessor = make_preprocessor(
                X_resampled,
                [f for f in self.numerical_features_ if f in set(X_resampled.columns)],
                [f for f in self.categorical_features_ if f in set(X_resampled.columns)],
            )

            X_preprocessed = preprocessor.fit_transform(X_resampled)
            X_preprocessed = check_array(
                X_preprocessed,
                dtype=np.float64,
                accept_sparse=False,
            )

            estimator = self._fit_estimator(
                X_preprocessed,
                y_resampled,
                b=self.b,
                c=c_resampled,
            )

            local_support = np.asarray(estimator.support_, dtype=int)
            global_support = tuple(col_names.take(local_support).tolist())

            return True, col_names, global_support, None

        except Exception as exc:

            return False, None, None, repr(exc)
    
    def _make_base_estimator(self, b, c):
        if self.estimator is None:
            return L0L2Classifier(
                b=b,
                c=c,
                fit_intercept=self.fit_intercept,
                time_limit=self.mosek_time_limit,
                mosek_log=self.mosek_log,
            )

        estimator = clone(self.estimator)

        params = estimator.get_params()
        updates = {}

        if "b" in params:
            updates["b"] = b
        if "c" in params:
            updates["c"] = c
        if "fit_intercept" in params:
            updates["fit_intercept"] = self.fit_intercept
        if "mosek_log" in params:
            updates["mosek_log"] = self.mosek_log
        if "time_limit" in params:
            updates["time_limit"] = self.mosek_time_limit

        if updates:
            estimator.set_params(**updates)

        return estimator

    def _preprocess(self, X, fit: bool=False):
        if fit:
            self.preprocessor_ = make_preprocessor(
                X,
                [f for f in self.numerical_features_ if f in set(X.columns)],
                [f for f in self.categorical_features_ if f in set(X.columns)],
            )
            X = self.preprocessor_.fit_transform(X)
        else:
            X = self.preprocessor_.transform(X)

        return check_array(X, dtype=np.float64, accept_sparse=False)

    def _sample_columns(self, rng, n_samples: int, n_features: int) -> np.ndarray:
        if self.resampling in {"none", "bootstrap_rows", "subsample_rows"}:
            return np.arange(n_features)
        
        if self.resampling == "auto" and n_features <= 1000:
            return np.arange(n_features)

        if self.resampling in {"subsample_columns", "subsample_both", "auto"}:
            size = resolve_subsample_size(self.n_column_subsamples, n_features)
            return np.sort(
                resample(
                    np.arange(n_features),
                    replace=False,
                    n_samples=size,
                    random_state=rng,
                )
            )

        raise RuntimeError("Invalid resampling state.")

    def _sample_rows(self, rng, y, n_samples: int, n_features: int) -> np.ndarray:
        if self.resampling in {"none", "subsample_columns"}:
            return np.arange(n_samples)

        if self.resampling in {"subsample_rows", "subsample_both"}:
            size = resolve_subsample_size(self.n_row_subsamples, n_samples)
            return resample(
                np.arange(n_samples),
                replace=False,
                n_samples=size,
                stratify=y,
                random_state=rng,
            )

        if self.resampling == "bootstrap_rows":
            return resample(
                np.arange(n_samples),
                replace=True,
                n_samples=n_samples,
                stratify=y,
                random_state=rng,
            )

        if self.resampling == "auto":
            if n_samples <= 10000:
                return resample(
                    np.arange(n_samples),
                    replace=True,
                    n_samples=n_samples,
                    stratify=y,
                    random_state=rng,
                )

            size = resolve_subsample_size(self.n_row_subsamples, n_samples)
            return resample(
                np.arange(n_samples),
                replace=False,
                n_samples=size,
                stratify=y,
                random_state=rng,
            )

        raise RuntimeError("Invalid resampling state.")
    
    def _top_vif_within_budget(self, vif: pd.Series) -> pd.Index:
        vif = vif.sort_values(ascending=False)

        selected = []
        used_budget = 0.0

        for feature in vif.index:
            cost = float(self.c_.loc[feature])
            if used_budget + cost <= float(self.b):
                selected.append(feature)
                used_budget += cost

        return pd.Index(selected)

    def _validate_params(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        if self.b is None:
            raise TypeError("b must not be None.")

        if not np.isfinite(self.b):
            raise ValueError("b must be finite.")

        if self.b <= 0:
            raise ValueError("b must be positive.")

        if self.c is None:
            self.c_ = pd.Series(1.0, index=X.columns, dtype=np.float64)
        else:
            c = np.asarray(self.c, dtype=np.float64)

            if c.shape != (n_features,):
                raise ValueError(
                    f"c must have shape (n_features,). "
                    f"Got c.shape={c.shape}, n_features={n_features}."
                )

            if not np.all(np.isfinite(c)):
                raise ValueError("All entries of c must be finite.")

            if np.any(c < 0):
                raise ValueError("All entries of c must be nonnegative.")
            
            self.c_ = pd.Series(c, index=X.columns, dtype=np.float64)

        valid_resampling = {
            "auto",
            "bootstrap_rows",
            "subsample_rows",
            "subsample_columns",
            "subsample_both",
            "none",
        }

        if self.resampling not in valid_resampling:
            raise ValueError(
                f"resampling must be one of {sorted(valid_resampling)}. "
                f"Got {self.resampling!r}."
            )

        uses_row_subsampling = self.resampling in {
            "subsample_rows",
            "subsample_both",
        } or (self.resampling == "auto" and n_samples > 10000)

        uses_column_subsampling = self.resampling in {
            "subsample_columns",
            "subsample_both",
        } or (self.resampling == "auto" and n_features > 1000)

        for name, value, max_value, is_used in [
            ("n_row_subsamples", self.n_row_subsamples, n_samples, uses_row_subsampling),
            ("n_column_subsamples", self.n_column_subsamples, n_features, uses_column_subsampling),
        ]:
            if isinstance(value, (int, np.integer)):
                if value <= 0:
                    raise ValueError(f"{name} must be positive.")
                if is_used and value > max_value:
                    raise ValueError(f"{name} cannot exceed {max_value}.")
            elif isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    raise ValueError(f"{name} must be finite.")
                if not 0 < value <= 1:
                    raise ValueError(f"{name} as a float must be in (0, 1].")
            else:
                raise TypeError(f"{name} must be an int or float.")

        if uses_row_subsampling:
            classes = np.unique(y)
            row_subsample_size = resolve_subsample_size(
                self.n_row_subsamples,
                n_samples,
            )
            if row_subsample_size < len(classes):
                raise ValueError(
                    "n_row_subsamples must be at least the number of classes "
                    "when using stratified row subsampling."
                )

        if not isinstance(self.n_resamples, (int, np.integer)):
            raise TypeError("n_resamples must be an integer.")

        if self.n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")

        if self.aggregation not in {"MSF", "VIF"}:
            raise ValueError("aggregation must be either 'MSF' or 'VIF'.")

        if not np.isfinite(self.vif_threshold):
            raise ValueError("vif_threshold must be finite.")

        if not 0 <= self.vif_threshold <= 1:
            raise ValueError("vif_threshold must be between 0 and 1.")

        if self.mosek_time_limit is not None:
            if not np.isfinite(self.mosek_time_limit):
                raise ValueError("mosek_time_limit must be finite or None.")

            if self.mosek_time_limit <= 0:
                raise ValueError("mosek_time_limit must be positive or None.")
        
        if self.total_time_limit is not None:
            if not np.isfinite(self.total_time_limit):
                raise ValueError("total_time_limit must be finite or None.")

            if self.total_time_limit <= 0:
                raise ValueError("total_time_limit must be positive or None.")

        if not isinstance(self.max_consecutive_failures, (int, np.integer)):
            raise TypeError("max_consecutive_failures must be an integer.")

        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive.")
        
        if self.n_jobs is not None:
            if not isinstance(self.n_jobs, (int, np.integer)):
                raise TypeError("n_jobs must be an integer or None.")
            if self.n_jobs == 0:
                raise ValueError("n_jobs cannot be 0.")

    def _validate_X_predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ResampledL0L2Classifier requires X to be a pandas DataFrame at "
                "prediction time."
            )
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")

        fitted_columns = pd.Index(self.feature_names_in_)
        incoming_columns = pd.Index(X.columns)

        missing = fitted_columns.difference(incoming_columns)
        extra = incoming_columns.difference(fitted_columns)
        if len(missing) or len(extra):
            raise ValueError(
                "X must contain exactly the columns seen during fitting. "
                f"Missing columns: {list(missing)!r}. Extra columns: {list(extra)!r}."
            )

        return X

    def _validate_X_y(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ResampledL0L2Classifier requires X to be a pandas DataFrame so "
                "feature identities can be tracked by column name."
            )
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")

        y = np.asarray(y)
        check_consistent_length(X, y)
        check_classification_targets(y)
        if len(np.unique(y)) != 2:
            raise ValueError("ResampledL0L2Classifier only supports binary classification.")

        return X, y
