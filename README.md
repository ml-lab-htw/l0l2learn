# l0l2learn

Feature selection for logistic regression using mixed-integer conic optimization. Unlike Lasso-based approaches, `l0l2learn` directly optimizes feature subsets under explicit cardinality or budget constraints.

## Overview

`l0l2learn` is a Python package that provides sklearn-style estimators for cardinality- and budget-constrained feature selection in logistic regression. The package currently includes:

- **L0L2Classifier**: L0-constrained L2-regularized logistic regression
- **ResampledL0L2Classifier**: resampling-based feature selection with frequency-based aggregation to improve the selection stability

## Installation

To install the package, use the following command:

```sh
pip install l0l2learn
```

Please check the [MOSEK website](https://www.mosek.com/) to request and set up a license for the conic solver.

## Quick Start

### Feature Selection Without Resampling

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from l0l2learn import L0L2Classifier


X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y
)

clf = L0L2Classifier(
    b=3,
    lambd=1.0
)

clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)

print("ROC AUC:      ", roc_auc_score(y_test, y_proba[:, 1]))
print("Coefficients: ", clf.coef_)
print("Intercept:    ", clf.intercept_)
print("Support:      ", clf.support_)
```

### Feature Selection With Resampling


```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from l0l2learn import ResampledL0L2Classifier


X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y
)

clf = ResampledL0L2Classifier(
    b=3,
    param_grid={"lambd": [1.0]},
    n_resamples=3
)

clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)

print("ROC AUC:      ", roc_auc_score(y_test, y_proba[:, 1]))
print("Coefficients: ", clf.coef_)
print("Intercept:    ", clf.intercept_)
print("Support:      ", clf.support_)
print("VIFs:         ", clf.variable_inclusion_frequencies_[clf.support_])
print("MSFs:         ", clf.model_selection_frequencies_)
```

## Hyperparameters

### Feature Costs

Feature-specific costs can be supplied through `c`:

```python
clf = L0L2Classifier(c=[1, 2, 5])
```

The optimization then accounts for some variables to be more expensive than others.

### Feature Selection Budget

The feature selection budget is controlled through `b`:

```python
clf = L0L2Classifier(b=5)
```

When all feature costs are equal to one (the default), `b` directly controls the maximum number of selected features.

### L2 Regularization

The L2 regularization strength is given by `lambd`:

```python
clf = L0L2Classifier(lambd=0.1)
```

Larger values can attenuate overfitting and increase robustness.

### Number of Resamples

`n_resamples` determines how many resampled models are fitted:

```python
clf = ResampledL0L2Classifier(b=5, n_resamples=99)
```

Larger values can improve frequency estimates but increase runtime.

### Other Hyperparameters

#### L0L2Classifier

- **`fit_intercept`**: Whether an intercept term is included in the logistic regression model.

- **`time_limit`**: Maximum runtime in seconds for the optimization problem.

- **`mosek_log`**: Enables printing of MOSEK solver output.

#### ResampledL0L2Classifier

- **`resampling`**: Controls whether and how rows, columns, both, or neither are resampled.

- **`n_row_subsamples`**: Number or fraction of observations used during row subsampling.

- **`n_column_subsamples`**: Number or fraction of features used during column subsampling.

- **`aggregation`**: Whether model selection or variable inclusion frequencies are used for aggregation.

- **`vif_threshold`**: Minimum variable inclusion frequency required when using `aggregation="VIF"`.

- **`estimator`**: Alternative base estimator used instead of the default `L0L2Classifier`.

- **`param_grid`**: Hyperparameter grid used for cross-validation when tuning `lambd`.

- **`cv`**: Cross-validation strategy used for hyperparameter tuning.

- **`scoring`**: Scoring metric used to select the best hyperparameter configuration.

- **`numerical_features`**: Specifies which DataFrame columns should be treated as numerical features.

- **`categorical_features`**: Specifies which DataFrame columns should be treated as categorical features.

- **`fit_intercept`**: Whether an intercept term is included in the logistic regression model.

- **`mosek_time_limit`**: Maximum runtime in seconds for each individual optimization problem.

- **`total_time_limit`**: Maximum runtime in seconds for the complete resampling procedure.

- **`max_consecutive_failures`**: Stops resampling if too many consecutive model fits fail.

- **`mosek_log`**: Enables printing of MOSEK solver output.

- **`n_jobs`**: Number of parallel workers used during resampling.

- **`random_state`**: Controls the randomness of resampling and cross-validation procedures.

## License

This project is licensed under the [MIT License](https://github.com/ml-lab-htw/l0l2learn/blob/main/LICENSE). See the `LICENSE` file for details.

## Authors

- Ricardo Knauer (HTW Berlin)